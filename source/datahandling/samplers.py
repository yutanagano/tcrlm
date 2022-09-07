'Custom sampler classes'


import random
from torch.utils.data import Sampler
from typing import Callable


class SortedBatchSampler(Sampler):
    '''
    Custom batch sampler which optimises batching according to some sorting
    metric, defined by sort_a.
    '''

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        sort_a: Callable,
        shuffle: bool = False
    ) -> None:
        self._num_samples = num_samples
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._len = (num_samples + batch_size - 1) // batch_size
        self._sort_a = sort_a


    def __iter__(self) -> list:
        superbatched = self._batch(self._get_indices(), self._batch_size * 100)

        for sb in superbatched:
            sorted_sb = sorted(
                sb,
                key=self._sort_a
            )

            batched = self._batch(sorted_sb, self._batch_size)
            if self._shuffle:
                random.shuffle(batched)
            
            for b in batched:
                yield b


    def __len__(self) -> int:
        return self._len


    def _batch(self, data: list, batch_size: int) -> list:
        '''
        Take a list of items and segment it into smaller lists of size <=
        batch_size where all segments are of length batch_size until the very
        last batch which may be smaller if the length of the dataset is not
        exactly divisible.
        '''
        num_batches = (len(data) + batch_size - 1) // batch_size
        batched = []
        for i in range(num_batches):
            batched.append(
                data[
                    i*batch_size:
                    min((i+1)*batch_size,len(data))
                ]
            )
        return batched


    def _get_indices(self) -> list:
        if self._shuffle:
            return random.sample(
                range(self._num_samples),
                k=self._num_samples
            )
        
        return range(self._num_samples)