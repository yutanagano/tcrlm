import pytest
import torch
from source.data_handling.datasets import Cdr3FineTuneDataset
from source.data_handling.dataloaders import Cdr3FineTuneDataLoader


@pytest.fixture(scope='module')
def get_path_to_mock_csv(get_path_to_project):
    return 'tests/data/mock_labelled_data.csv'


@pytest.fixture(scope='module')
def instantiate_dataset(get_path_to_mock_csv):
    dataset = Cdr3FineTuneDataset(data=get_path_to_mock_csv)
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
        for x_1a_batch, x_1b_batch, x_2a_batch, x_2b_batch, y_batch \
            in dataloader:
            assert(
                type(x_1a_batch) == type(x_1b_batch) == type(x_2a_batch) \
                    == type(x_2b_batch) == type(y_batch) == torch.Tensor
            )
            assert(
                x_1a_batch.size(0) == x_1b_batch.size(0) == x_2a_batch.size(0) \
                == x_2b_batch.size(0) == y_batch.size(0)
            )
            assert(x_1a_batch.size(0) in (5,1))
            assert(
                x_1a_batch.dim() == x_1b_batch.dim() == \
                x_2a_batch.dim() == x_2b_batch.dim() == 2
            )
            assert(y_batch.dim() == 1)


def test_dataloader_with_distributed_sampler(instantiate_dataset):
    dataloader = Cdr3FineTuneDataLoader(
        dataset=instantiate_dataset,
        batch_size=5,
        shuffle=True,
        distributed=True,
        num_replicas=2,
        rank=0
    )

    # Ensure that the dataloader length is half (becuase num_replicas = 2) of
    # the length of the dataset, divided by 5 (because batch_size = 5). The
    # 'plus four' is to ensure that the integer division returns the ceiling of
    # the division, and not the floor.
    assert(len(dataloader) == (len(instantiate_dataset) + 9) // (2 * 5))

    for i in range(2):
        for x_1a_batch, x_1b_batch, x_2a_batch, x_2b_batch, y_batch \
            in dataloader:
            assert(
                type(x_1a_batch) == type(x_1b_batch) == type(x_2a_batch) \
                    == type(x_2b_batch) == type(y_batch) == torch.Tensor
            )
            assert(
                x_1a_batch.size(0) == x_1b_batch.size(0) == x_2a_batch.size(0) \
                == x_2b_batch.size(0) == y_batch.size(0)
            )
            assert(x_1a_batch.size(0) in (5,3))
            assert(
                x_1a_batch.dim() == x_1b_batch.dim() == \
                x_2a_batch.dim() == x_2b_batch.dim() == 2
            )
            assert(y_batch.dim() == 1)


# Negative tests
def test_incorrect_dataset_type():
    dataset = torch.utils.data.Dataset()
    with pytest.raises(AssertionError):
        dataloader = Cdr3FineTuneDataLoader(dataset, 5)