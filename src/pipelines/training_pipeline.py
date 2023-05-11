from abc import ABC, abstractmethod
import argparse
import json
import os
import pandas as pd
from pathlib import Path
from random import randint
import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.nn import Module
from torch.utils.data import DataLoader


class TrainingPipeline(ABC):
    description = "Abstract base class for training pipelines."

    def __init__(self) -> None:
        self.world_size = torch.cuda.device_count()
    
    def run_from_clargs(self) -> None:
        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument(
            "-d",
            "--working-directory",
            help="Path to tcr_embedder project working directory.",
        )
        parser.add_argument(
            "config_path", help="Path to the training run config json file."
        )
        args = parser.parse_args()

        if args.working_directory is None:
            wd = Path.cwd()
        else:
            wd = Path(args.working_directory).resolve()

        assert wd.is_dir()

        with open(args.config_path, "r") as f:
            config = json.load(f)

        self.main(wd=wd, config=config)

    def main(self, wd: Path, config: dict) -> None:
        self.wd = wd
        self.config = config

        config["data"]["dataloader"]["config"][
            "batch_size"
        ] //= self.world_size  # Correct for DDP

        print(f"Commencing training on {self.world_size} CUDA device(s)...")
        port = randint(10000, 60000)
        print(f"Coordinating on port {port}...")
        mp.spawn(self.proc, args=(port,), nprocs=self.world_size)

    def proc(self, rank: int, port: int) -> None:
        self.ddp_setup(port, rank)

        # Load training objects
        TrainingPipeline.proc_print("Loading training objects...", rank)
        model, train_dl, valid_dl, loss_fns, optimiser = self.training_obj_factory(
            self.config, rank
        )

        # Evaluate model at pre-SimC learning state
        TrainingPipeline.proc_print("Evaluating pre-trained model state...", rank)
        valid_metrics = self.valid_func(
            model=model,
            dl=valid_dl,
            loss_fns=loss_fns,
            rank=rank,
        )
        TrainingPipeline.metric_feedback(valid_metrics, rank)

        metric_log = {0: {"loss": None, "lr": None, **valid_metrics}}

        # Go through epochs of training
        for epoch in range(1, self.config["n_epochs"] + 1):
            TrainingPipeline.proc_print(f"Starting epoch {epoch}...", rank)
            train_dl.set_epoch(epoch)

            TrainingPipeline.proc_print("Training...", rank)
            train_metrics = self.train_func(
                model=model,
                dl=train_dl,
                loss_fns=loss_fns,
                optimiser=optimiser,
                rank=rank,
            )
            TrainingPipeline.metric_feedback(train_metrics, rank)

            TrainingPipeline.proc_print("Validating...", rank)
            valid_metrics = self.valid_func(
                model=model,
                dl=valid_dl,
                loss_fns=loss_fns,
                rank=rank,
            )
            TrainingPipeline.metric_feedback(valid_metrics, rank)

            metric_log[epoch] = {**train_metrics, **valid_metrics}

        # Save results
        if rank == 0:
            print("Saving results...")
            self.save(model=model.module.embedder, log=metric_log)

        self.ddp_cleanup(rank)
        TrainingPipeline.proc_print("Done!", rank)

    def ddp_setup(self, port: int, rank: int) -> None:
        TrainingPipeline.proc_print("Setting up DDP process...", rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        init_process_group(backend="nccl", rank=rank, world_size=self.world_size)
        TrainingPipeline.proc_print("DDP process started.", rank)

    def ddp_cleanup(self, rank: int) -> None:
        TrainingPipeline.proc_print("Closing DDP process...", rank)
        destroy_process_group()
        TrainingPipeline.proc_print("DDP process closed.", rank)

    def save(self, model: Module, log: dict) -> None:
        model_saves_dir = self.wd / "model_saves"
        try:
            model_saves_dir.mkdir()
        except FileExistsError:
            pass

        save_name = self.config["model"]["config"]["name"]

        try:
            (model_saves_dir / save_name).mkdir()
        except FileExistsError:
            suffix_int = 1
            new_save_name = f"{save_name}_{suffix_int}"
            done = False
            while not done:
                try:
                    (model_saves_dir / new_save_name).mkdir()
                    save_name = new_save_name
                    done = True
                except FileExistsError:
                    suffix_int += 1
                    new_save_name = f"{save_name}_{suffix_int}"
        
        save_dir = model_saves_dir / save_name

        # Save model
        model.cpu()
        torch.save(model.state_dict(), save_dir / "state_dict.pt")

        # Save log
        pd.DataFrame.from_dict(log, orient="index").to_csv(
            save_dir / "log.csv", index_label="epoch"
        )

        # Save config
        with open(save_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=4)

    @staticmethod
    def proc_print(msg: str, rank: int) -> None:
        print(f"[{rank}] {msg}")

    @staticmethod
    def metric_feedback(metrics: dict, rank: int) -> None:
        for metric in metrics:
            TrainingPipeline.proc_print(f"{metric}: {metrics[metric]}", rank)

    @staticmethod
    @abstractmethod
    def training_obj_factory(config: dict, rank: int) -> tuple:
        """
        Factory function that should use the config file and the current
        process's rank to return the following objects:

        - Model object
        - Training dataloader
        - Validation dataloader
        - Loss function(s)
        - Optimiser
        """

    @staticmethod
    @abstractmethod
    def train_func(
        model: Module,
        dl: DataLoader,
        loss_fns: tuple,
        optimiser,
        rank: int
    ) -> dict:
        """
        Run a training epoch, and return a dictionary containing the average
        epoch loss and learning rate.
        """

    @staticmethod
    @abstractmethod
    def valid_func(model: Module, dl: DataLoader, loss_fns: tuple, rank: int) -> dict:
        """
        Run a validation epoch, and return some summary statistics as a
        dictionary.
        """