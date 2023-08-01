from torch import Tensor
from torch.nn.utils import rnn
from torch.utils.data import DataLoader
from typing import Tuple, Union


class TcrDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, collate_fn=self.collate_fn)

    def collate_fn(self, batch) -> Union[Tuple[Tensor], Tensor]:
        if self._batch_has_multielement_samples(batch):
            return self._collate_multielement_batch(batch)

        return self._pad_batch_of_sequences(batch)

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)

    def _batch_has_multielement_samples(self, batch) -> bool:
        first_element = batch[0]
        return isinstance(first_element, list) or isinstance(first_element, tuple)
    
    def _collate_multielement_batch(self, batch) -> Tuple[Tensor]:
        batch_per_element = zip(*batch)
        padded_batch_per_element = map(self._pad_batch_of_sequences, batch_per_element)

        return tuple(padded_batch_per_element)
    
    def _pad_batch_of_sequences(self, batch) -> Tensor:
        return rnn.pad_sequence(sequences=batch, batch_first=True, padding_value=0)