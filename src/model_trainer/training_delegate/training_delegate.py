from abc import ABC, abstractmethod

from src.model_trainer.training_object_collection import TrainingObjectCollection


class TrainingDelegate(ABC):
    @abstractmethod
    def run_training_epoch_and_return_metrics_for(
        self, training_object_collection: TrainingObjectCollection
    ) -> dict:
        pass

    @abstractmethod
    def validate_and_return_metrics_for(
        self, training_object_collection: TrainingObjectCollection
    ) -> dict:
        pass
