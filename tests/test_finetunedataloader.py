import pytest
import torch
from torch.utils.data.distributed import DistributedSampler
from source.data_handling import Cdr3FineTuneDataset, Cdr3FineTuneDataLoader


@pytest.fixture(scope='module')
def get_path_to_mock_csv(get_path_to_project):
    return 'tests/data/mock_labelled_data.csv'


@pytest.fixture(scope='module')
def instantiate_dataset(get_path_to_mock_csv):
    dataset = Cdr3FineTuneDataset(path_to_csv=get_path_to_mock_csv)
    yield dataset


@pytest.fixture(scope='module')
def instantiate_dataloader(instantiate_dataset):
    dataloader = Cdr3FineTuneDataLoader(
        dataset=instantiate_dataset,
        batch_size=5
    )
    yield dataloader


# Positive tests
def test_dataloader(instantiate_dataloader):
    dataloader = instantiate_dataloader
    
    for i in range(2):
        for x_1_batch, x_2_batch, y_batch in dataloader:
            assert(
                type(x_1_batch) == type(x_2_batch) == type(y_batch) == \
                    torch.Tensor
            )
            assert(x_1_batch.size(0) == x_2_batch.size(0) == y_batch.size(0))
            assert(x_1_batch.size(0) in (5,1))
            assert(x_1_batch.dim() == x_2_batch.dim() == 2)
            assert(y_batch.dim() == 1)


def test_dataloader_with_distributed_sampler(instantiate_dataset):
    test_sampler = DistributedSampler(
        dataset=instantiate_dataset,
        num_replicas=2,
        rank=0,
        shuffle=True,
        seed=0
    )
    dataloader = Cdr3FineTuneDataLoader(
        dataset=instantiate_dataset,
        batch_size=5,
        distributed_sampler=test_sampler
    )

    # Ensure that the dataloader length is half (becuase num_replicas = 2) of
    # the length of the dataset, divided by 5 (because batch_size = 5). The
    # 'plus four' is to ensure that the integer division returns the ceiling of
    # the division, and not the floor.
    assert(len(dataloader) == (len(instantiate_dataset) + 4) // (2 * 5))

    for i in range(2):
        for x_1_batch, x_2_batch, y_batch in dataloader:
            assert(
                type(x_1_batch) == type(x_2_batch) == type(y_batch) == \
                    torch.Tensor
            )
            assert(x_1_batch.size(0) == x_2_batch.size(0) == y_batch.size(0))
            assert(x_1_batch.size(0) in (5,3))
            assert(x_1_batch.dim() == x_2_batch.dim() == 2)
            assert(y_batch.dim() == 1)


# Negative tests
def test_incorrect_dataset_type():
    dataset = torch.utils.data.Dataset()
    with pytest.raises(AssertionError):
        dataloader = Cdr3FineTuneDataLoader(dataset, 5)


def test_set_both_distributed_sampler_shuffle(instantiate_dataset):
    test_sampler = DistributedSampler(
        dataset=instantiate_dataset,
        num_replicas=2,
        rank=0,
        shuffle=True,
        seed=0
    )
    with pytest.raises(RuntimeError):
        dataloader = Cdr3FineTuneDataLoader(
            dataset=instantiate_dataset,
            batch_size=5,
            shuffle=True,
            distributed_sampler=test_sampler
        )

def test_set_bad_distributed_sampler(instantiate_dataset):
    with pytest.raises(AssertionError):
        dataloader = Cdr3FineTuneDataLoader(
            dataset=instantiate_dataset,
            batch_size=5,
            distributed_sampler=5
        )