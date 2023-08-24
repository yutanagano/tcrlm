import json
import os
import pandas as pd
from pathlib import Path
import random
import re
import torch
from torch import distributed
from torch import multiprocessing
from typing import Optional

from src.model_trainer.config_reader import ConfigReader
from src.model.bert import Bert
from src.model_trainer.training_object_collection import TrainingObjectCollection


class ModelTrainer:
    def __init__(self, config: dict) -> None:
        self._config_reader = ConfigReader(config)
        self._training_delegate = self._config_reader.get_training_delegate()
        self._num_gpus_available = torch.cuda.device_count()

    def train(self, working_directory: Optional[Path] = None) -> None:
        self._set_working_directory(working_directory)
        self._launch_training_processes()

    def _set_working_directory(self, working_directory: Optional[Path] = None) -> None:
        if working_directory is not None:
            self._working_directory = working_directory
        else:
            self._working_directory = Path.cwd()

    def _launch_training_processes(self) -> None:
        port = random.randint(10000, 60000)
        print(
            f"Launching {self._num_gpus_available} training processes on port {port}..."
        )
        multiprocessing.spawn(
            self._training_process, args=(port,), nprocs=self._num_gpus_available
        )

    def _training_process(self, gpu_index: int, port: int) -> None:
        self._setup_ddp(gpu_index, port)

        training_object_collection = self._instantiate_training_objects(gpu_index)

        self._print_with_process_id("Evaluating pre-trained model state...", gpu_index)
        valid_metrics = self._validate_and_return_metrics_for(
            training_object_collection
        )
        self._print_metrics_with_process_id(valid_metrics, gpu_index)

        metric_log = {0: {"loss": None, "lr": None, **valid_metrics}}

        for epoch in range(1, self._get_num_epochs() + 1):
            self._print_with_process_id(f"Starting epoch {epoch}...", gpu_index)

            training_object_collection.training_dataloader.set_epoch(epoch)

            train_metrics = self._run_training_epoch_and_return_metrics_for(
                training_object_collection
            )
            self._print_metrics_with_process_id(train_metrics, gpu_index)

            valid_metrics = self._validate_and_return_metrics_for(
                training_object_collection
            )
            self._print_metrics_with_process_id(valid_metrics, gpu_index)

            metric_log[epoch] = {**train_metrics, **valid_metrics}

        current_process_using_first_gpu = gpu_index == 0
        if current_process_using_first_gpu:
            print("Saving results...")
            unwrapped_bert_model = training_object_collection.model.module.bert
            self._save_training_results(
                model=unwrapped_bert_model, metric_log=metric_log
            )

        self._clean_up_ddp()
        self._print_with_process_id("Done!", gpu_index)

    def _instantiate_training_objects(self, gpu_index: int) -> TrainingObjectCollection:
        return self._config_reader.get_training_object_collection_on_device(gpu_index)

    def _get_num_epochs(self) -> int:
        return self._config_reader.get_num_epochs()

    def _run_training_epoch_and_return_metrics_for(
        self, training_object_collection: TrainingObjectCollection
    ) -> dict:
        return self._training_delegate.run_training_epoch_and_return_metrics_for(
            training_object_collection
        )

    def _validate_and_return_metrics_for(
        self, training_object_collection: TrainingObjectCollection
    ) -> dict:
        return self._training_delegate.validate_and_return_metrics_for(
            training_object_collection
        )

    def _save_training_results(self, model: Bert, metric_log: dict) -> None:
        model_save_dir = self._make_model_save_dir_and_return_path()

        self._save_model(model, model_save_dir)
        self._save_metric_log(metric_log, model_save_dir)
        self._save_config(model_save_dir)

    def _make_model_save_dir_and_return_path(self) -> Path:
        model_saves_parent_dir = self._working_directory / "model_saves"
        self._create_directory_if_not_already_present(model_saves_parent_dir)
        model_save_directory_name = self._get_model_name_without_special_characters()
        model_save_dir = model_saves_parent_dir / model_save_directory_name

        try:
            model_save_dir.mkdir()
        except FileExistsError:
            model_save_dir = self._make_renamed_version_of_dir_and_return_path(
                model_save_dir
            )

        return model_save_dir

    def _create_directory_if_not_already_present(self, path_to_directory: Path) -> None:
        try:
            path_to_directory.mkdir()
        except FileExistsError:
            pass

    def _get_model_name_without_special_characters(self) -> str:
        model_name = self._config_reader.get_model_name()
        without_whitespace = re.sub(r"\s+", "_", model_name)
        non_alphanumerics_removed = re.sub(r"\W", "", without_whitespace)
        return non_alphanumerics_removed

    def _make_renamed_version_of_dir_and_return_path(
        self, path_to_original_dir: Path
    ) -> Path:
        dir_name = path_to_original_dir.name
        parent_dir = path_to_original_dir.parent
        suffix_int = 1

        while True:
            full_dir_path = parent_dir / f"{dir_name}_{suffix_int}"
            try:
                full_dir_path.mkdir()
                return full_dir_path
            except FileExistsError:
                suffix_int += 1

    def _save_model(self, model: Bert, model_save_dir: Path) -> None:
        model_moved_to_cpu = model.cpu()
        torch.save(model_moved_to_cpu.state_dict(), model_save_dir / "state_dict.pt")

    def _save_metric_log(self, metric_log: dict, model_save_dir: Path) -> None:
        metric_log_as_dataframe = pd.DataFrame.from_dict(metric_log, orient="index")
        metric_log_as_dataframe.to_csv(model_save_dir / "log.csv", index_label="epoch")

    def _save_config(self, model_save_dir: Path) -> None:
        config = self._config_reader.get_config()

        with open(model_save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)

    def _print_metrics_with_process_id(self, metrics: dict, gpu_index: int) -> None:
        for metric, value in metrics.items():
            self._print_with_process_id(f"{metric}: {value}", gpu_index)

    def _print_with_process_id(self, msg: str, gpu_index: int) -> None:
        print(f"[{gpu_index}] {msg}")

    def _setup_ddp(self, gpu_index: int, port: int) -> None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        distributed.init_process_group(
            backend="nccl", rank=gpu_index, world_size=self._num_gpus_available
        )

    def _clean_up_ddp(self) -> None:
        distributed.destroy_process_group()
