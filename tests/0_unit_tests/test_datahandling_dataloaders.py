import pytest
import source.datahandling.dataloaders as dataloaders
import source.datahandling.datasets as datasets
import torch
from torch.utils.data.distributed import DistributedSampler


@pytest.fixture(scope='module')
def pretrain_dataset():
    dataset = datasets.Cdr3PretrainDataset(
        data='tests/resources/data/mock_unlabelled.csv',
    )
    return dataset


@pytest.fixture(scope='module')
def finetune_dataset():
    dataset = datasets.Cdr3FineTuneDataset(
        data='tests/resources/data/mock_labelled.csv'
    )
    return dataset


@pytest.fixture(scope='function')
def sorted_batch_sampler(pretrain_dataset):
    sampler = dataloaders.SortedBatchSampler(
        num_samples=len(pretrain_dataset),
        batch_size=5,
        sort_a=pretrain_dataset.get_length
    )
    return sampler


@pytest.mark.parametrize(
    ('input_seq','expected'),
    (
        ('CAST', torch.tensor([1,0,15,16])),
        ('MESH', torch.tensor([10,3,15,6])),
        ('WILL', torch.tensor([18,7,9,9]))
    )
)
def test_tokenise(input_seq, expected):
    assert torch.equal(dataloaders.tokenise(input_seq), expected)


class TestDefineSampling:
    def test_vanilla(self, pretrain_dataset):
        result = dataloaders.define_sampling(
            dataset=pretrain_dataset,
            batch_size=10,
            shuffle=True,
            distributed=False,
            batch_optimisation=False
        )
        expected = {
            'batch_size': 10,
            'shuffle': True,
            'sampler': None,
            'batch_sampler': None
        }

        assert result == expected

    
    def test_distributed(self, pretrain_dataset):
        result = dataloaders.define_sampling(
            dataset=pretrain_dataset,
            batch_size=10,
            shuffle=False,
            distributed=True,
            batch_optimisation=False,
            num_replicas=2,
            rank=0
        )

        assert result['batch_size'] == 10
        assert result['shuffle'] is None
        assert type(result['sampler']) == DistributedSampler
        assert result['sampler'].dataset == pretrain_dataset
        assert result['sampler'].num_replicas == 2
        assert result['sampler'].rank == 0
        assert result['sampler'].shuffle == False
        assert result['sampler'].seed == 0
        assert result['batch_sampler'] is None


    def test_batch_optimisation(self, pretrain_dataset):
        sort_a = lambda x: x

        result = dataloaders.define_sampling(
            dataset=pretrain_dataset,
            batch_size=10,
            shuffle=True,
            distributed=False,
            batch_optimisation=True,
            sort_a=sort_a
        )

        assert result['batch_size'] == 1
        assert result['shuffle'] is None
        assert result['sampler'] is None
        assert type(result['batch_sampler']) == dataloaders.SortedBatchSampler
        assert result['batch_sampler']._num_samples == 29
        assert result['batch_sampler']._batch_size == 10
        assert result['batch_sampler']._sort_a == sort_a
        assert result['batch_sampler']._shuffle == True
    

    def test_distributed_no_num_replicas_rank(self, pretrain_dataset):
        with pytest.raises(RuntimeError):
            dataloaders.define_sampling(
                dataset=pretrain_dataset,
                batch_size=10,
                shuffle=False,
                distributed=True,
                batch_optimisation=False
            )


    def test_distributed_batch_optimisation(self, pretrain_dataset):
        with pytest.raises(RuntimeError):
            dataloaders.define_sampling(
                dataset=pretrain_dataset,
                batch_size=10,
                shuffle=True,
                distributed=True,
                batch_optimisation=True,
                num_replicas=2,
                rank=0
            )


    def test_batch_optimisation_no_sort_a(self, pretrain_dataset):
        with pytest.raises(RuntimeError):
            dataloaders.define_sampling(
                dataset=pretrain_dataset,
                batch_size=10,
                shuffle=False,
                distributed=False,
                batch_optimisation=True
            )


class TestSortedBatchSampler:
    def test_iter(self, sorted_batch_sampler):
        for i, batch in enumerate(sorted_batch_sampler):
            if i == 0:
                assert batch[0] == 0
                continue
            if i == 5:
                assert batch[-1] == 28
    

    def test_len(self, sorted_batch_sampler):
        assert len(sorted_batch_sampler) == 6


class TestTcrDataLoader:
    def test_init_bad_dataset_type(self):
        with pytest.raises(AssertionError):
            dataloaders.TcrDataLoader(dataset=[0,1,2], batch_size=10)


class TestCdr3PretrainDataLoader:
    def test_init_bad_dataset_type(self):
        with pytest.raises(AssertionError):
            dataloaders.Cdr3PretrainDataLoader(dataset=[0,1,2], batch_size=10)


    @pytest.mark.parametrize(
        ('shuffle'), (False, True)
    )
    def test_iter(self, pretrain_dataset, shuffle):
        dataloader = dataloaders.Cdr3PretrainDataLoader(
            dataset=pretrain_dataset,
            batch_size=5,
            shuffle=shuffle
        )
        
        for src_batch, tgt_batch in dataloader:
            assert type(src_batch) == type(tgt_batch) == torch.Tensor
            assert src_batch.size() == tgt_batch.size()
            assert src_batch.size(0) in (4, 5)
            assert src_batch.size(1) >= 10 and src_batch.size(1) <= 20
            assert src_batch.dim() == 2


    @pytest.mark.parametrize(
        ('shuffle'), (False, True)
    )
    def test_iter_distributed(self, pretrain_dataset, shuffle):
        dataloader = dataloaders.Cdr3PretrainDataLoader(
            dataset=pretrain_dataset,
            batch_size=5,
            shuffle=shuffle,
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
            assert src_batch.size(0) in (4, 5)
            assert src_batch.size(1) >= 10 and src_batch.size(1) <= 20
            assert src_batch.dim() == 2

    
    @pytest.mark.parametrize(
        ('shuffle'), (False, True)
    )
    def test_iter_batch_optimisation(self, pretrain_dataset, shuffle):
        dataloader = dataloaders.Cdr3PretrainDataLoader(
            dataset=pretrain_dataset,
            batch_size=5,
            shuffle=shuffle,
            batch_optimisation=True
        )

        min_batch_seq_len_encountered = 999

        for src_batch, tgt_batch in dataloader:
            assert type(src_batch) == type(tgt_batch) == torch.Tensor
            assert src_batch.size() == tgt_batch.size()
            assert src_batch.size(0) in (4, 5)
            assert src_batch.size(1) >= 10 and src_batch.size(1) <= 20
            assert src_batch.dim() == 2
            min_batch_seq_len_encountered = min(
                src_batch.size(1),
                min_batch_seq_len_encountered
            )
        
        assert min_batch_seq_len_encountered == 12


class TestCdr3FineTuneDataLoader:
    def test_init_bad_dataset_type(self):
        with pytest.raises(AssertionError):
            dataloaders.Cdr3FineTuneDataLoader(dataset=[0,1,2], batch_size=10)


    @pytest.mark.parametrize(
        ('shuffle'), (False, True)
    )
    def test_iter(self, finetune_dataset, shuffle):
        dataloader = dataloaders.Cdr3FineTuneDataLoader(
            dataset=finetune_dataset,
            batch_size=5,
            shuffle=shuffle
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
    

    @pytest.mark.parametrize(
        ('shuffle'), (False, True)
    )
    def test_iter_distributed(self, finetune_dataset, shuffle):
        dataloader = dataloaders.Cdr3FineTuneDataLoader(
            dataset=finetune_dataset,
            batch_size=5,
            shuffle=shuffle,
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