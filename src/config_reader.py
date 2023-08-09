import pandas as pd
from pathlib import Path
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from types import ModuleType

from src.training_object_collection import TrainingObjectCollection
from src.model.trainable_model import TrainableModel
from src.model.bert import Bert
import src.model.trainable_model as trainable_model_module
from src.model.token_embedder.token_embedder import TokenEmbedder
import src.model.token_embedder as token_embedder_module
from src.model.self_attention_stack import SelfAttentionStack
import src.model.self_attention_stack as self_attention_stack_module
from src.model.mlm_token_prediction_projector import MlmTokenPredictionProjector
import src.model.mlm_token_prediction_projector as mlm_token_prediction_projector_module
from src.model.vector_representation_delegate import VectorRepresentationDelegate
import src.model.vector_representation_delegate as vector_representation_delegate_module
from src.data.tcr_dataloader import TcrDataLoader
from src.data.tokeniser.tokeniser import Tokeniser
import src.data.tokeniser as tokeniser_module
from src.data.tcr_dataset import TcrDataset
from src.batch_collator.batch_collator import BatchCollator
import src.batch_collator as batch_collator_module
import src.metric as metric_module
from src.optim import AdamWithScheduling


class ConfigReader:
    def __init__(self, config: dict) -> None:
        self._config = config

    def get_training_object_collection_on_device(self, device: torch.device) -> TrainingObjectCollection:
        if not isinstance(device, torch.device):
            device = torch.device(device)

        model = self._instantiate_trainable_ddp_on_device(device)
        training_dataloader = self._instantiate_training_dataloader()
        validation_dataloader = self._instantiate_validation_dataloader()
        loss_functions = self._instantiate_loss_functions()
        optimiser = self._instantiate_optimiser_for_model(model)

        return TrainingObjectCollection(
            model=model,
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            loss_functions=loss_functions,
            optimiser=optimiser,
            device=device
        )
    
    def instantiate_bert_on_device(self, device: torch.device) -> Bert:
        token_embedder = self._instantiate_token_embedder()
        self_attention_stack = self._instantiate_self_attention_stack()
        mlm_token_prediction_projector = self._instantiate_mlm_token_prediction_projector()
        vector_representation_delegate = self._instantiate_vector_representation_delegate_for_self_attention_stack(self_attention_stack)

        bert = Bert(token_embedder=token_embedder,
                    self_attention_stack=self_attention_stack,
                    mlm_token_prediction_projector=mlm_token_prediction_projector,
                    vector_representation_delegate=vector_representation_delegate)
        bert_on_device = bert.to(device)

        return bert_on_device
    
    def get_num_epochs(self) -> int:
        return self._config["num_epochs"]
    
    def get_model_name(self) -> str:
        return self._config["model"]["name"]
    
    def get_config(self) -> dict:
        return self._config
    
    def _instantiate_trainable_ddp_on_device(self, device: torch.device) -> DistributedDataParallel:
        trainable_model = self._instantiate_trainable_model_on_device(device)
        return DistributedDataParallel(trainable_model)
    
    def _instantiate_trainable_model_on_device(self, device: torch.device) -> TrainableModel:
        trainable_model_wrapper_class_name = self._config["model"]["trainable_model"]["class"]
        TrainableModelWrapperClass = getattr(trainable_model_module, trainable_model_wrapper_class_name)
        bert = self.instantiate_bert_on_device(device)
        bert = self._load_bert_with_pretrained_parameters_if_available(bert)
        return TrainableModelWrapperClass(bert)
    
    def _load_bert_with_pretrained_parameters_if_available(self, bert: Bert) -> Bert:
        path_to_pretrained_state_dict_as_str = self._config["model"]["path_to_pretrained_state_dict"]

        if path_to_pretrained_state_dict_as_str is not None:
            state_dict = torch.load(Path(path_to_pretrained_state_dict_as_str))
            bert.load_state_dict(state_dict)

        return bert
    
    def _instantiate_token_embedder(self) -> TokenEmbedder:
        config = self._config["model"]["token_embedder"]
        return self._instantiate_object_from_module_using_config(token_embedder_module, config)
    
    def _instantiate_self_attention_stack(self) -> SelfAttentionStack:
        config = self._config["model"]["self_attention_stack"]
        return self._instantiate_object_from_module_using_config(self_attention_stack_module, config)
    
    def _instantiate_mlm_token_prediction_projector(self) -> MlmTokenPredictionProjector:
        config = self._config["model"]["mlm_token_prediction_projector"]
        return self._instantiate_object_from_module_using_config(mlm_token_prediction_projector_module, config)
    
    def _instantiate_vector_representation_delegate_for_self_attention_stack(self, self_attention_stack: SelfAttentionStack) -> VectorRepresentationDelegate:
        config = self._config["model"]["vector_representation_delegate"]
        class_name = config["class"]
        initargs = config["initargs"]
        VectorRepresentationDelegateClass = getattr(vector_representation_delegate_module, class_name)
        return VectorRepresentationDelegateClass(self_attention_stack=self_attention_stack, **initargs)
    
    def _instantiate_training_dataloader(self) -> TcrDataLoader:
        path_to_training_data_csv_as_str = self._config["data"]["path_to_training_data"]
        dataloader_initargs = self._config["data"]["dataloader"]["initargs"]

        tokeniser = self._instantiate_tokeniser()
        dataset = self._instantiate_dataset(Path(path_to_training_data_csv_as_str), tokeniser)
        batch_collator = self._instantiate_batch_collator_for_tokeniser(tokeniser)

        return TcrDataLoader(
            dataset=dataset,
            sampler=DistributedSampler(dataset, shuffle=True),
            collate_fn=batch_collator.collate_fn,
            **dataloader_initargs
        )
    
    def _instantiate_validation_dataloader(self) -> TcrDataLoader:
        path_to_validation_data_csv_as_str = self._config["data"]["path_to_validation_data"]
        dataloader_initargs = self._config["data"]["dataloader"]["initargs"]

        tokeniser = self._instantiate_tokeniser()
        dataset = self._instantiate_dataset(Path(path_to_validation_data_csv_as_str), tokeniser)
        batch_collator = self._instantiate_batch_collator_for_tokeniser(tokeniser)

        return TcrDataLoader(
            dataset=dataset,
            collate_fn=batch_collator.collate_fn,
            **dataloader_initargs
        )
    
    def _instantiate_tokeniser(self) -> Tokeniser:
        config = self._config["data"]["tokeniser"]
        return self._instantiate_object_from_module_using_config(tokeniser_module, config)
    
    def _instantiate_dataset(self, path_to_training_data_csv: Path, tokeniser: Tokeniser) -> TcrDataset:
        df = pd.read_csv(path_to_training_data_csv)
        return TcrDataset(data=df, tokeniser=tokeniser)

    def _instantiate_batch_collator_for_tokeniser(self, tokeniser: Tokeniser) -> BatchCollator:
        config = self._config["data"]["batch_collator"]
        class_name = config["class"]
        initargs = config["initargs"]
        BatchCollatorClass = getattr(batch_collator_module, class_name)
        return BatchCollatorClass(tokeniser=tokeniser, **initargs)
    
    def _instantiate_loss_functions(self) -> dict:
        loss_function_configs = self._config["loss"]
        loss_functions = {
            name: self._instantiate_object_from_module_using_config(metric_module, config)
            for name, config in loss_function_configs.items()
        }
        return loss_functions
    
    def _instantiate_optimiser_for_model(self, model: DistributedDataParallel) -> AdamWithScheduling:
        config = self._config["optimiser"]
        initargs = config["initargs"]
        return AdamWithScheduling(params=model.parameters(), **initargs)

    def _instantiate_object_from_module_using_config(self, module: ModuleType, config: dict) -> any:
        class_name = config["class"]
        initargs = config["initargs"]
        Class = getattr(module, class_name)
        return Class(**initargs)