from ..class_method_metaclass import ClassMethodMeta
import argparse
import json
import os
import pandas as pd
from pathlib import Path
from random import randint
import re
import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.nn import Module
from torch.utils.data import DataLoader


class TrainingPipeline(metaclass=ClassMethodMeta):
    DESCRIPTION = "Abstract base class for training pipelines."

    def run_from_clargs(cls) -> None:
        parser = argparse.ArgumentParser(description=cls.DESCRIPTION)
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

        cls.main(wd=wd, config=config)

    def main(cls, wd: Path, config: dict) -> None:
        world_size = torch.cuda.device_count()
        config = cls.correct_batch_size(config, world_size)

        print(f"Commencing training on {world_size} CUDA device(s)...")
        cls.launch_training_processes(wd, config, world_size)

    @staticmethod
    def correct_batch_size(config: dict, world_size: int) -> dict:
        config["data"]["dataloader"]["config"][
            "batch_size"
        ] //= world_size  # Correct for DDP
        return config

    def launch_training_processes(cls, wd: Path, config: dict, world_size: int) -> None:
        port = randint(10000, 60000)
        print(f"Coordinating on port {port}...")
        mp.spawn(cls.proc, args=(port, wd, config, world_size), nprocs=world_size)

    def proc(cls, rank: int, port: int, wd: Path, config: dict, world_size: int) -> None:
        cls.ddp_setup(port, rank, world_size)

        cls.proc_print("Loading training objects...", rank)
        model, train_dl, valid_dl, loss_fns, optimiser = cls.training_obj_factory(
            config, rank
        )

        cls.proc_print("Evaluating pre-trained model state...", rank)
        valid_metrics = cls.valid_func(
            model=model,
            dl=valid_dl,
            loss_fns=loss_fns,
            rank=rank,
        )
        cls.metric_feedback(valid_metrics, rank)

        metric_log = {0: {"loss": None, "lr": None, **valid_metrics}}

        # Go through epochs of training
        for epoch in range(1, config["n_epochs"] + 1):
            cls.proc_print(f"Starting epoch {epoch}...", rank)
            train_dl.set_epoch(epoch)

            cls.proc_print("Training...", rank)
            train_metrics = cls.train_func(
                model=model,
                dl=train_dl,
                loss_fns=loss_fns,
                optimiser=optimiser,
                rank=rank,
            )
            cls.metric_feedback(train_metrics, rank)

            cls.proc_print("Validating...", rank)
            valid_metrics = cls.valid_func(
                model=model,
                dl=valid_dl,
                loss_fns=loss_fns,
                rank=rank,
            )
            cls.metric_feedback(valid_metrics, rank)

            metric_log[epoch] = {**train_metrics, **valid_metrics}

        # Save results
        if rank == 0:
            print("Saving results...")
            cls.save(wd=wd, config=config, model=model.module.embedder, log=metric_log)

        cls.ddp_cleanup(rank)
        cls.proc_print("Done!", rank)

    def ddp_setup(cls, port: int, rank: int, world_size: int) -> None:
        cls.proc_print("Setting up DDP process...", rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        cls.proc_print("DDP process started.", rank)

    def ddp_cleanup(cls, rank: int) -> None:
        cls.proc_print("Closing DDP process...", rank)
        destroy_process_group()
        cls.proc_print("DDP process closed.", rank)

    @staticmethod
    def save(wd: Path, config: dict, model: Module, log: dict) -> None:
        model_saves_dir = wd / "model_saves"
        try:
            model_saves_dir.mkdir()
        except FileExistsError:
            pass

        save_name = re.sub(r"\s", "_", config["model"]["config"]["name"])
        save_name = re.sub(r"\W", "", save_name)

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
            json.dump(config, f, indent=4)

    @staticmethod
    def proc_print(msg: str, rank: int) -> None:
        print(f"[{rank}] {msg}")

    @staticmethod
    def metric_feedback(metrics: dict, rank: int) -> None:
        for metric in metrics:
            TrainingPipeline.proc_print(f"{metric}: {metrics[metric]}", rank)

    @staticmethod
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
        raise NotImplementedError()

    @staticmethod
    def train_func(
        model: Module, dl: DataLoader, loss_fns: tuple, optimiser, rank: int
    ) -> dict:
        """
        Run a training epoch, and return a dictionary containing the average
        epoch loss and learning rate.
        """
        raise NotImplementedError()

    @staticmethod
    def valid_func(model: Module, dl: DataLoader, loss_fns: tuple, rank: int) -> dict:
        """
        Run a validation epoch, and return some summary statistics as a
        dictionary.
        """
        raise NotImplementedError()
