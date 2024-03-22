import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from src.optim import AdamWithScheduling


class TrainingObjectCollection:
    def __init__(
        self,
        model: DistributedDataParallel,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
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
