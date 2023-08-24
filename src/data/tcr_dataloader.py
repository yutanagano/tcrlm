from torch import LongTensor
from torch.utils.data import DataLoader
from torch.nn import utils
from typing import Iterable

from src.data.tokeniser.token_indices import DefaultTokenIndex


def default_collate_fn(batch: Iterable[LongTensor]) -> LongTensor:
    return utils.rnn.pad_sequence(
        sequences=batch, batch_first=True, padding_value=DefaultTokenIndex.NULL
    )


class TcrDataLoader(DataLoader):
    def __init__(self, *args, collate_fn=default_collate_fn, **kwargs):
        super().__init__(*args, collate_fn=collate_fn, **kwargs)

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)
