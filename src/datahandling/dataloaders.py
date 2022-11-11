'''
Custom dataloader classes.
'''


from src.datahandling.datasets import TCRDataset
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Union


class TCRDataLoader(DataLoader):
    '''
    Base dataloader class.
    '''

    def __init__(
        self,
        dataset: TCRDataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        **kwargs
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            **kwargs
        )


    def collate_fn(self, batch) -> Union[Tuple[Tensor], Tensor]:
        'Pad and batch tokenised TCRs.'

        elem = batch[0]

        if isinstance(elem, list) or isinstance(elem, tuple):
            return tuple(
                map(
                    lambda x: pad_sequence(
                        sequences=x,
                        batch_first=True,
                        padding_value=0
                    ),
                    zip(*batch)
                )
            )

        return pad_sequence(
            sequences=batch,
            batch_first=True,
            padding_value=0
        )