import torch
from torch.nn.parallel import DistributedDataParallel

from src.nn.data.tcr_dataloader import SingleDatasetDataLoader
from src.nn.optim import AdamWithScheduling


class TrainingObjectCollection:
    def __init__(
        self,
        model: DistributedDataParallel,
        training_dataloader: SingleDatasetDataLoader,
        validation_dataloader: SingleDatasetDataLoader,
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
