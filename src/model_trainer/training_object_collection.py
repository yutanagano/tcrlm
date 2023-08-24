import torch
from torch.nn.parallel import DistributedDataParallel

from src.data.tcr_dataloader import TcrDataLoader
from src.model_trainer.optim import AdamWithScheduling


class TrainingObjectCollection:
    def __init__(
        self,
        model: DistributedDataParallel,
        training_dataloader: TcrDataLoader,
        validation_dataloader: TcrDataLoader,
        loss_functions: dict,
        optimiser: AdamWithScheduling,
        device: torch.device,
    ):
        self.model = model
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.loss_functions = loss_functions
        self.optimiser = optimiser
        self.device = device
