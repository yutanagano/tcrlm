from torch import Tensor
from torch.nn.utils import rnn
from torch.utils.data import DataLoader
from typing import Tuple, Union


class TcrDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)