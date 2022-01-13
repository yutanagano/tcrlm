import os
import pytest
import torch
from source.data_handling import CDR3Dataset, CDR3DataLoader


@pytest.fixture(scope='module')
def get_path_to_mock_csv(get_path_to_project):
    return os.path.join(get_path_to_project, 'tests/data/mock_data.csv')


@pytest.fixture(scope='module')
def instantiate_dataset(get_path_to_mock_csv):
    dataset = CDR3Dataset(path_to_csv=get_path_to_mock_csv)
    yield dataset


@pytest.fixture(scope='module')
def instantiate_dataloader(instantiate_dataset):
    dataloader = CDR3DataLoader(dataset=instantiate_dataset,
                                batch_size=5)
    yield dataloader


# Positive tests
def test_dataloader(instantiate_dataloader):
    dataloader = instantiate_dataloader
    src_batch, tgt_batch = next(iter(dataloader))

    assert(type(src_batch) == type(tgt_batch) == torch.Tensor)
    assert(src_batch.size() == tgt_batch.size())
    assert(src_batch.size(0) == 5)
    assert(src_batch.size(1) == 3 or src_batch.size(1) == 4)
    assert(src_batch.dim() == 2)