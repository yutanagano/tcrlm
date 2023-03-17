import argparse
import datetime
import json
import os
import pandas as pd
from pathlib import Path
import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.nn import Module
from typing import Callable


class TrainingPipeline:
    def __init__(
        self,
        description: str,
        training_obj_factory: Callable,
        train_func: Callable,
        valid_func: Callable,
    ) -> None:
        self.description = description
        self.world_size = torch.cuda.device_count()
        self.training_obj_factory = training_obj_factory
        self.train_func = train_func
        self.valid_func = valid_func

    def run_from_clargs(self) -> None:
        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument(
            "-d",
            "--working-directory",
            help="Path to tcr_embedder project working directory.",
        )
        parser.add_argument("-n", "--name", help="Name of the training run.")
        parser.add_argument(
            "config_path", help="Path to the training run config json file."
        )
        args = parser.parse_args()

        if args.working_directory is None:
            wd = Path.cwd()
        else:
            wd = Path(args.working_directory).resolve()

        if args.name is None:
            name = datetime.now().strftime(r"%Y%m%d-%H%M%S")

        assert wd.is_dir()

        with open(args.config_path, "r") as f:
            config = json.load(f)

        self.main(wd=wd, name=name, config=config)

    def main(self, wd: Path, name: str, config: dict) -> None:
        self.wd = wd
        self.name = name
        self.config = config

        config["data"]["dataloader"]["config"][
            "batch_size"
        ] //= self.world_size  # Correct for DDP

        print(f"Commencing training on {self.world_size} CUDA device(s)...")
        mp.spawn(self.proc, nprocs=self.world_size)

    def proc(self, rank: int) -> None:
        self.ddp_setup(rank)

        # Load training objects
        print("Loading training objects...")
        model, train_dl, valid_dl, loss_fns, optimiser = self.training_obj_factory(
            self.config, rank
        )

        # Evaluate model at pre-SimC learning state
        print("Evaluating pre-trained model state...")
        valid_metrics = self.valid_func(
            model=model,
            dl=valid_dl,
            loss_fns=loss_fns,
            rank=rank,
        )
        TrainingPipeline.metric_feedback(valid_metrics)

        metric_log = {0: {"loss": None, "lr": None, **valid_metrics}}

        # Go through epochs of training
        for epoch in range(1, self.config["n_epochs"] + 1):
            print(f"Starting epoch {epoch}...")
            train_dl.sampler.set_epoch(epoch)

            print("Training...")
            train_metrics = self.train_func(
                model=model,
                dl=train_dl,
                loss_fns=loss_fns,
                optimiser=optimiser,
                rank=rank,
            )
            TrainingPipeline.metric_feedback(train_metrics)

            print("Validating...")
            valid_metrics = self.valid_func(
                model=model,
                dl=valid_dl,
                loss_fns=loss_fns,
                rank=rank,
            )
            TrainingPipeline.metric_feedback(valid_metrics)

            metric_log[epoch] = {**train_metrics, **valid_metrics}

        # Save results
        if rank == 0:
            print("Saving results...")
            self.save(model=model.module.embedder, log=metric_log)

        self.ddp_cleanup()
        print("Done!")

    def ddp_setup(self, rank: int) -> None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        init_process_group(backend="nccl", rank=rank, world_size=self.world_size)

    def ddp_cleanup(self) -> None:
        destroy_process_group()

    def save(self, model: Module, log: dict) -> None:
        model_saves_dir = self.wd / "model_saves"
        try:
            model_saves_dir.mkdir()
        except FileExistsError:
            pass

        try:
            (model_saves_dir / self.name).mkdir()
        except FileExistsError:
            suffix_int = 1
            new_save_name = f"{self.name}_{suffix_int}"
            done = False
            while not done:
                try:
                    (model_saves_dir / new_save_name).mkdir()
                    self.name = new_save_name
                    done = True
                except FileExistsError:
                    suffix_int += 1
                    new_save_name = f"{self.name}_{suffix_int}"
        save_dir = model_saves_dir / self.name

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
    def metric_feedback(metrics: dict) -> None:
        for metric in metrics:
            print(f"{metric}: {metrics[metric]}")
