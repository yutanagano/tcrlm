from libtcrlm.bert import Bert
from libtcrlm.config_reader import ConfigReader
from libtcrlm.tokeniser import Tokeniser
import pandas as pd
from pathlib import Path
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, Dataset, DataLoader
from types import ModuleType

import src.performance_measure as performance_measure_module
import src.trainable_model as trainable_model_module
import src.data.batch_collator as batch_collator_module
import src.training_delegate as training_delegate_module
import src.data.dataset as dataset_module

from src.data.dataloader import SingleDatasetDataLoader, DoubleDatasetDataLoader
from src.trainable_model import TrainableModel
from src.data.batch_collator import BatchCollator
from src.optim import AdamWithScheduling
from src.training_delegate import TrainingDelegate

from src.config_reader.training_object_collection import TrainingObjectCollection


class TrainingConfigReader:
    def __init__(self, config: dict) -> None:
        self._config_reader = ConfigReader(config)

    def get_model_name(self) -> str:
        return self._config_reader.get_model_name()

    def get_config(self) -> dict:
        return self._config_reader.get_config()

    def get_num_epochs(self) -> int:
        return self.get_config()["num_epochs"]

    def get_training_delegate(self) -> TrainingDelegate:
        config = self.get_config()["training_delegate"]
        return self._get_object_from_module_using_config(
            training_delegate_module, config
        )

    def get_training_object_collection_on_device(
        self, device: torch.device
    ) -> TrainingObjectCollection:
        if not isinstance(device, torch.device):
            device = torch.device(device)

        model = self._get_trainable_ddp_on_device(device)
        training_dataloader = self._get_training_dataloader_on_device(device)
        validation_dataloader = self._get_validation_dataloader_on_device(device)
        loss_functions = self._get_loss_functions()
        optimiser = self._get_optimiser_for_model(model)

        return TrainingObjectCollection(
            model=model,
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            loss_functions=loss_functions,
            optimiser=optimiser,
            device=device,
        )

    def _get_trainable_ddp_on_device(
        self, device: torch.device
    ) -> DistributedDataParallel:
        trainable_model = self._get_trainable_model_on_device(device)
        return DistributedDataParallel(trainable_model)

    def _get_trainable_model_on_device(self, device: torch.device) -> TrainableModel:
        trainable_model_wrapper_class_name = self.get_config()["model"]["trainable_model"][
            "class"
        ]
        TrainableModelWrapperClass = getattr(
            trainable_model_module, trainable_model_wrapper_class_name
        )
        bert = self._config_reader.get_bert()
        bert = self._load_bert_with_pretrained_parameters_if_available(bert)
        bert = bert.to(device)
        return TrainableModelWrapperClass(bert)

    def _load_bert_with_pretrained_parameters_if_available(self, bert: Bert) -> Bert:
        path_to_pretrained_state_dict_as_str = self.get_config()["model"][
            "path_to_pretrained_state_dict"
        ]

        if path_to_pretrained_state_dict_as_str is not None:
            state_dict = torch.load(Path(path_to_pretrained_state_dict_as_str))
            bert.load_state_dict(state_dict)

        return bert

    def _get_training_dataloader_on_device(self, device: torch.device) -> DataLoader:
        data_loader_class = self.get_config()["data"]["training_data"]["dataloader"]["class"]

        if data_loader_class == "SingleDatasetDataLoader":
            dataloader = self._get_single_dataset_training_dataloader_on_device(device)
        elif data_loader_class == "DistributedDoubleDatasetDataLoader":
            dataloader = self._get_double_dataset_training_dataloader_on_device(device)
        else:
            raise ValueError(f"Unrecognised dataloader class: {data_loader_class}")

        return dataloader

    def _get_single_dataset_training_dataloader_on_device(
        self, device: torch.device
    ) -> SingleDatasetDataLoader:
        path_to_training_data_csv_as_str = self.get_config()["data"]["training_data"][
            "csv_paths"
        ][0]
        dataloader_initargs = self.get_config()["data"]["training_data"]["dataloader"][
            "initargs"
        ]

        tokeniser = self._config_reader.get_tokeniser()
        dataset = self._get_training_dataset(Path(path_to_training_data_csv_as_str))
        batch_collator = self._get_batch_collator_with_tokeniser(tokeniser)

        return SingleDatasetDataLoader(
            dataset=dataset,
            sampler=DistributedSampler(dataset, shuffle=True),
            batch_collator=batch_collator,
            device=device,
            **dataloader_initargs,
        )

    def _get_double_dataset_training_dataloader_on_device(
        self, device: torch.device
    ) -> DoubleDatasetDataLoader:
        paths_to_training_data_csvs_as_str = self.get_config()["data"]["training_data"][
            "csv_paths"
        ]
        dataloader_initargs = self.get_config()["data"]["training_data"]["dataloader"][
            "initargs"
        ]

        tokeniser = self._config_reader.get_tokeniser()
        dataset_1 = self._get_training_dataset(Path(paths_to_training_data_csvs_as_str[0]))
        dataset_2 = self._get_training_dataset(Path(paths_to_training_data_csvs_as_str[1]))
        batch_collator = self._get_batch_collator_with_tokeniser(tokeniser)

        return DoubleDatasetDataLoader(
            dataset_1=dataset_1,
            dataset_2=dataset_2,
            batch_collator=batch_collator,
            device=device,
            **dataloader_initargs,
        )

    def _get_validation_dataloader_on_device(
        self, device: torch.device
    ) -> DataLoader:
        path_to_validation_data_csv_as_str = self.get_config()["data"]["validation_data"][
            "csv_paths"
        ][0]
        dataloader_initargs = self.get_config()["data"]["validation_data"]["dataloader"][
            "initargs"
        ]

        tokeniser = self._config_reader.get_tokeniser()
        dataset = self._get_validation_dataset(Path(path_to_validation_data_csv_as_str))
        batch_collator = self._get_batch_collator_with_tokeniser(tokeniser)

        return SingleDatasetDataLoader(
            dataset=dataset,
            batch_collator=batch_collator,
            device=device,
            **dataloader_initargs,
        )

    def _get_training_dataset(self, path_to_training_data_csv: Path) -> Dataset:
        training_dataset_config = self.get_config()["data"]["training_data"]["dataset"]
        TrainingDatasetClass = getattr(dataset_module, training_dataset_config["class"])

        df = pd.read_csv(path_to_training_data_csv)

        for column in (
            "TRAV",
            "CDR3A",
            "TRAJ",
            "TRBV",
            "CDR3B",
            "TRBJ",
            "Epitope",
            "MHCA",
            "MHCB",
        ):
            if column not in df:
                df[column] = pd.NA

        return TrainingDatasetClass(df, **training_dataset_config["initargs"])
    
    def _get_validation_dataset(self, path_to_validation_data_csv: Path) -> Dataset:
        validation_dataset_config = self.get_config()["data"]["validation_data"]["dataset"]
        ValidationDatasetClass = getattr(dataset_module, validation_dataset_config["class"])

        df = pd.read_csv(path_to_validation_data_csv)

        for column in (
            "TRAV",
            "CDR3A",
            "TRAJ",
            "TRBV",
            "CDR3B",
            "TRBJ",
            "Epitope",
            "MHCA",
            "MHCB",
        ):
            if column not in df:
                df[column] = pd.NA

        return ValidationDatasetClass(df, **validation_dataset_config["initargs"])

    def _get_batch_collator_with_tokeniser(self, tokeniser: Tokeniser) -> BatchCollator:
        config = self.get_config()["data"]["batch_collator"]
        class_name = config["class"]
        initargs = config["initargs"]
        BatchCollatorClass = getattr(batch_collator_module, class_name)
        return BatchCollatorClass(tokeniser=tokeniser, **initargs)

    def _get_loss_functions(self) -> dict:
        loss_function_configs = self.get_config()["loss"]
        loss_functions = {
            name: self._get_object_from_module_using_config(
                performance_measure_module, config
            )
            for name, config in loss_function_configs.items()
        }
        return loss_functions

    def _get_optimiser_for_model(
        self, model: DistributedDataParallel
    ) -> AdamWithScheduling:
        config = self.get_config()["optimiser"]
        initargs = config["initargs"]
        return AdamWithScheduling(params=model.parameters(), **initargs)

    def _get_object_from_module_using_config(
        self, module: ModuleType, config: dict
    ) -> any:
        class_name = config["class"]
        initargs = config["initargs"]
        Class = getattr(module, class_name)
        return Class(**initargs)
