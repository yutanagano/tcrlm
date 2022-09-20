import pytest
from source.datahandling import dataloaders, datasets, samplers, tokenisers
import torch
from torch.utils.data import (RandomSampler, SequentialSampler, BatchSampler)
from torch.utils.data.distributed import DistributedSampler


@pytest.fixture
def pretrain_dataset():
    dataset = datasets.Cdr3PretrainDataset(
        data='tests/resources/data/mock_unlabelled.csv',
        tokeniser=tokenisers.AaTokeniser(len_tuplet=1)
    )
    return dataset


@pytest.fixture
def finetune_dataset():
    dataset = datasets.Cdr3FineTuneDataset(
        data='tests/resources/data/mock_labelled.csv',
        tokeniser=tokenisers.AaTokeniser(len_tuplet=1)
    )
    return dataset


class TestTcrDataLoader:
    def test_init_vanilla(self, pretrain_dataset):
        dataloader = dataloaders.TcrDataLoader(
            dataset=pretrain_dataset,
            batch_size=10
        )

        assert dataloader.dataset == pretrain_dataset
        assert dataloader.batch_size == 10
        assert type(dataloader.sampler) == SequentialSampler
        assert dataloader.sampler.data_source == pretrain_dataset
        assert type(dataloader.batch_sampler) == BatchSampler
        assert dataloader.batch_sampler.sampler == dataloader.sampler
        assert dataloader.batch_sampler.batch_size == 10


    def test_init_shuffle(self, pretrain_dataset):
        dataloader = dataloaders.TcrDataLoader(
            dataset=pretrain_dataset,
            batch_size=10,
            shuffle=True
        )

        assert dataloader.dataset == pretrain_dataset
        assert dataloader.batch_size == 10
        assert type(dataloader.sampler) == RandomSampler
        assert dataloader.sampler.data_source == pretrain_dataset
        assert type(dataloader.batch_sampler) == BatchSampler
        assert dataloader.batch_sampler.sampler == dataloader.sampler
        assert dataloader.batch_sampler.batch_size == 10


    def test_init_num_workers(self, pretrain_dataset):
        dataloader = dataloaders.TcrDataLoader(
            dataset=pretrain_dataset,
            batch_size=10,
            num_workers=5
        )

        assert dataloader.dataset == pretrain_dataset
        assert dataloader.batch_size == 10
        assert type(dataloader.sampler) == SequentialSampler
        assert dataloader.sampler.data_source == pretrain_dataset
        assert type(dataloader.batch_sampler) == BatchSampler
        assert dataloader.batch_sampler.sampler == dataloader.sampler
        assert dataloader.batch_sampler.batch_size == 10
        assert dataloader.num_workers == 5


    def test_init_collate_fn(self, pretrain_dataset):
        collate_fn = lambda x: x

        dataloader = dataloaders.TcrDataLoader(
            dataset=pretrain_dataset,
            batch_size=10,
            collate_fn=collate_fn
        )

        assert dataloader.dataset == pretrain_dataset
        assert dataloader.batch_size == 10
        assert type(dataloader.sampler) == SequentialSampler
        assert dataloader.sampler.data_source == pretrain_dataset
        assert type(dataloader.batch_sampler) == BatchSampler
        assert dataloader.batch_sampler.sampler == dataloader.sampler
        assert dataloader.batch_sampler.batch_size == 10
        assert dataloader.collate_fn == collate_fn


    @pytest.mark.parametrize(
        'shuffle', (True, False)
    )
    def test_init_distributed(self, pretrain_dataset, shuffle):
        dataloader = dataloaders.TcrDataLoader(
            dataset=pretrain_dataset,
            batch_size=10,
            shuffle=shuffle,
            distributed=True,
            num_replicas=2,
            rank=0
        )

        assert dataloader.dataset == pretrain_dataset
        assert dataloader.batch_size == 10
        assert type(dataloader.sampler) == DistributedSampler
        assert dataloader.sampler.dataset == pretrain_dataset
        assert dataloader.sampler.num_replicas == 2
        assert dataloader.sampler.rank == 0
        assert dataloader.sampler.shuffle == shuffle
        assert dataloader.sampler.seed == 0
        assert type(dataloader.batch_sampler) == BatchSampler
        assert dataloader.batch_sampler.sampler == dataloader.sampler
        assert dataloader.batch_sampler.batch_size == 10


    @pytest.mark.parametrize(
        'shuffle', (True, False)
    )
    def test_init_batch_optimisation(self, pretrain_dataset, shuffle):
        sort_a = lambda x: x

        dataloader = dataloaders.TcrDataLoader(
            dataset=pretrain_dataset,
            batch_size=10,
            shuffle=shuffle,
            batch_optimisation=True,
            sort_a=sort_a
        )

        assert dataloader.dataset == pretrain_dataset
        assert dataloader.batch_size == None
        assert type(dataloader.sampler) == SequentialSampler
        assert dataloader.sampler.data_source == pretrain_dataset
        assert type(dataloader.batch_sampler) == samplers.SortedBatchSampler
        assert dataloader.batch_sampler._num_samples == len(pretrain_dataset)
        assert dataloader.batch_sampler._batch_size == 10
        assert dataloader.batch_sampler._sort_a == sort_a
        assert dataloader.batch_sampler._shuffle == shuffle


    def test_error_distributed_with_no_num_replicas(self, pretrain_dataset):
        with pytest.raises(RuntimeError):
            dataloaders.TcrDataLoader(
                dataset=pretrain_dataset,
                batch_size=10,
                distributed=True,
                rank=0
            )


    def test_error_distributed_with_no_rank(self, pretrain_dataset):
        with pytest.raises(RuntimeError):
            dataloaders.TcrDataLoader(
                dataset=pretrain_dataset,
                batch_size=10,
                distributed=True,
                num_replicas=2
            )

        
    def test_error_distributed_with_batch_optimisation(self, pretrain_dataset):
        with pytest.raises(RuntimeError):
            dataloaders.TcrDataLoader(
                dataset=pretrain_dataset,
                batch_size=10,
                distributed=True,
                num_replicas=2,
                rank=0,
                batch_optimisation=True,
                sort_a=(lambda x: x)
            )


    def test_error_batch_optimisation_with_no_sort_a(self, pretrain_dataset):
        with pytest.raises(RuntimeError):
            dataloaders.TcrDataLoader(
                dataset=pretrain_dataset,
                batch_size=10,
                batch_optimisation=True
            )


    def test_init_bad_dataset_type(self):
        with pytest.raises(AssertionError):
            dataloaders.TcrDataLoader(dataset=[0,1,2], batch_size=10)


class TestCdr3PretrainDataLoader:
    def test_iter(self, pretrain_dataset):
        dataloader = dataloaders.Cdr3PretrainDataLoader(
            dataset=pretrain_dataset,
            batch_size=5
        )
        
        for src_batch, tgt_batch in dataloader:
            assert type(src_batch) == type(tgt_batch) == torch.Tensor
            assert src_batch.size() == tgt_batch.size()
            assert src_batch.size(0) == 5
            assert src_batch.size(1) >= 10 and src_batch.size(1) <= 20
            assert src_batch.dim() == 2


    def test_iter_distributed(self, pretrain_dataset):
        dataloader = dataloaders.Cdr3PretrainDataLoader(
            dataset=pretrain_dataset,
            batch_size=5,
            distributed=True,
            num_replicas=2,
            rank=0
        )

        # Ensure that the dataloader length is half (becuase num_replicas = 2) 
        # of the length of the dataset, divided by 5 (because batch_size = 5).
        # The 'plus four' is to ensure that the integer division returns the
        # ceiling of the division, and not the floor.
        assert len(dataloader) == (len(pretrain_dataset) + 4) // (2 * 5)

        for src_batch, tgt_batch in dataloader:
            assert type(src_batch) == type(tgt_batch) == torch.Tensor
            assert src_batch.size() == tgt_batch.size()
            assert src_batch.size(0) == 5
            assert src_batch.size(1) >= 10 and src_batch.size(1) <= 20
            assert src_batch.dim() == 2

    
    def test_iter_batch_optimisation(self, pretrain_dataset):
        dataloader = dataloaders.Cdr3PretrainDataLoader(
            dataset=pretrain_dataset,
            batch_size=5,
            batch_optimisation=True
        )

        min_batch_seq_len_encountered = 999

        for src_batch, tgt_batch in dataloader:
            assert type(src_batch) == type(tgt_batch) == torch.Tensor
            assert src_batch.size() == tgt_batch.size()
            assert src_batch.size(0) == 5
            assert src_batch.size(1) >= 10 and src_batch.size(1) <= 20
            assert src_batch.dim() == 2
            min_batch_seq_len_encountered = min(
                src_batch.size(1),
                min_batch_seq_len_encountered
            )
        
        assert min_batch_seq_len_encountered == 12


class TestCdr3FineTuneDataLoader:
    def test_iter(self, finetune_dataset):
        dataloader = dataloaders.Cdr3FineTuneDataLoader(
            dataset=finetune_dataset,
            batch_size=5
        )

        for x_1a_batch, x_1b_batch, x_2a_batch, x_2b_batch, y_batch \
            in dataloader:
            assert type(x_1a_batch) == type(x_1b_batch) == type(x_2a_batch) \
                    == type(x_2b_batch) == type(y_batch) == torch.Tensor
            assert x_1a_batch.size(0) == x_1b_batch.size(0) == \
                x_2a_batch.size(0) == x_2b_batch.size(0) == y_batch.size(0)
            assert x_1a_batch.size(0) in (5,1)
            assert x_1a_batch.dim() == x_1b_batch.dim() == x_2a_batch.dim() \
                == x_2b_batch.dim() == 2
            assert y_batch.dim() == 1
    

    def test_iter_distributed(self, finetune_dataset):
        dataloader = dataloaders.Cdr3FineTuneDataLoader(
            dataset=finetune_dataset,
            batch_size=5,
            distributed=True,
            num_replicas=2,
            rank=0
        )

        # Ensure that the dataloader length is half (becuase num_replicas = 2) 
        # of the length of the dataset, divided by 5 (because batch_size = 5).
        # The 'plus four' is to ensure that the integer division returns the
        # ceiling of the division, and not the floor.
        assert len(dataloader) == (len(finetune_dataset) + 9) // (2 * 5)

        for x_1a_batch, x_1b_batch, x_2a_batch, x_2b_batch, y_batch \
            in dataloader:
            assert type(x_1a_batch) == type(x_1b_batch) == type(x_2a_batch) \
                    == type(x_2b_batch) == type(y_batch) == torch.Tensor
            assert x_1a_batch.size(0) == x_1b_batch.size(0) == \
                x_2a_batch.size(0) == x_2b_batch.size(0) == y_batch.size(0)
            assert x_1a_batch.size(0) in (5,3)
            assert x_1a_batch.dim() == x_1b_batch.dim() == x_2a_batch.dim() \
                == x_2b_batch.dim() == 2
            assert y_batch.dim() == 1