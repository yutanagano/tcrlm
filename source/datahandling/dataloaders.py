'Custom dataloader classes.'


from source.datahandling import datasets
from source.datahandling.samplers import SortedBatchSampler
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from typing import Callable, Tuple, Union


class TcrDataLoader(DataLoader):
    'Project custom base dataloader class.'

    def __init__(
        self,
        dataset: datasets.TcrDataset,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn: Union[Callable, None] = None,
        distributed: bool = False,
        num_replicas: Union[int, None] = None,
        rank: Union[int, None] = None,
        batch_optimisation: bool = False,
        sort_a: Union[Callable, None] = None
    ) -> None:
        assert issubclass(type(dataset), datasets.TcrDataset)

        sampling_settings = self._define_sampling(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            distributed=distributed,
            batch_optimisation=batch_optimisation,
            num_replicas=num_replicas,
            rank=rank,
            sort_a=sort_a
        )

        super(TcrDataLoader, self).__init__(
            dataset=dataset,
            batch_size=sampling_settings['batch_size'],
            shuffle=sampling_settings['shuffle'],
            sampler=sampling_settings['sampler'],
            batch_sampler=sampling_settings['batch_sampler'],
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )


    def _define_sampling(
        self,
        dataset: datasets.TcrDataset,
        batch_size: int,
        shuffle: bool,
        distributed: bool,
        batch_optimisation: bool,
        num_replicas: Union[int, None] = None,
        rank: Union[int, None] = None,
        sort_a: Union[Callable, None] = None
    ) -> dict:
        if not (distributed or batch_optimisation):
            return {
                'batch_size': batch_size,
                'shuffle': shuffle,
                'sampler': None,
                'batch_sampler': None
            }

        if distributed:
            if (num_replicas is None) or (rank is None):
                raise RuntimeError('Please specify num_replicas and rank.')
            if batch_optimisation:
                raise RuntimeError(
                    'Distributed sampling is mutually exclusive with batch '
                    'optimisation.'
                )
            return {
                'batch_size': batch_size,
                'shuffle': None,
                'sampler': DistributedSampler(
                    dataset=dataset,
                    num_replicas=num_replicas,
                    rank=rank,
                    shuffle=shuffle,
                    seed=0
                ),
                'batch_sampler': None
            }
        
        if batch_optimisation:
            # No need to check if batch_optimisation is also turned on because we
            # handled that case in the previous if block
            if sort_a is None:
                raise RuntimeError(
                    'Please specify the sorting algorithm for the batch optimiser.'
                    ' (sort_a)'
                )
            return {
                'batch_size': 1,
                'shuffle': None,
                'sampler': None,
                'batch_sampler': SortedBatchSampler(
                    num_samples=len(dataset),
                    batch_size=batch_size,
                    sort_a=sort_a,
                    shuffle=shuffle
                )
            }


class Cdr3PretrainDataLoader(TcrDataLoader):
    '''
    Dataloader for masked-residue modelling. Batch-optimisation adjust random
    batching to prioritise batching together CDR3s with similar lengths.
    '''

    def __init__(
        self,
        dataset: datasets.Cdr3PretrainDataset,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        distributed: bool = False,
        batch_optimisation: bool = False,
        num_replicas: Union[int, None] = None,
        rank: Union[int, None] = None
    ) -> None:
        assert type(dataset) == datasets.Cdr3PretrainDataset

        super(Cdr3PretrainDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            distributed=distributed,
            batch_optimisation=batch_optimisation,
            num_replicas=num_replicas,
            rank=rank,
            sort_a=dataset.get_length
        )

    
    def collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        'Tokenise and pad batch.'

        x_batch, y_batch = [], []
        for x_sample, y_sample in batch:
            x_batch.append(x_sample)
            y_batch.append(y_sample)

        x_batch = pad_sequence(
            sequences=x_batch,
            batch_first=True,
            padding_value=0
        )
        y_batch = pad_sequence(
            sequences=y_batch,
            batch_first=True,
            padding_value=0
        )

        return x_batch, y_batch


class Cdr3FineTuneDataLoader(TcrDataLoader):
    'Dataloader for epitope-matching finetuning.'

    def __init__(
        self,
        dataset: datasets.Cdr3FineTuneDataset,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        distributed: bool = False,
        num_replicas: Union[int, None] = None,
        rank: Union[int, None] = None
    ):
        assert(type(dataset) == datasets.Cdr3FineTuneDataset)

        super(Cdr3FineTuneDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            distributed=distributed,
            batch_optimisation=False,
            num_replicas=num_replicas,
            rank=rank
        )
    

    def collate_fn(
        self,
        batch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        'Tokenise and pad batch.'

        x_1a_batch, x_1b_batch, x_2a_batch, x_2b_batch, y_batch = \
            [], [], [], [], []
        
        for x_1a_sample, x_1b_sample, \
            x_2a_sample, x_2b_sample, y_sample in batch:
            x_1a_batch.append(x_1a_sample)
            x_1b_batch.append(x_1b_sample)
            x_2a_batch.append(x_2a_sample)
            x_2b_batch.append(x_2b_sample)
            y_batch.append(y_sample)
        
        x_1a_batch = pad_sequence(
            sequences=x_1a_batch,
            batch_first=True,
            padding_value=0
        )
        x_1b_batch = pad_sequence(
            sequences=x_1b_batch,
            batch_first=True,
            padding_value=0
        )
        x_2a_batch = pad_sequence(
            sequences=x_2a_batch,
            batch_first=True,
            padding_value=0
        )
        x_2b_batch = pad_sequence(
            sequences=x_2b_batch,
            batch_first=True,
            padding_value=0
        )
        y_batch = torch.tensor(y_batch, dtype=torch.long)

        return x_1a_batch, x_1b_batch, x_2a_batch, x_2b_batch, y_batch