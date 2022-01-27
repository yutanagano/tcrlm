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
def instantiate_dataloader_with_optim(instantiate_dataset):
    dataloader = CDR3DataLoader(
        dataset=instantiate_dataset,
        batch_size=5
    )
    yield dataloader


# Positive tests
def test_dataloader(instantiate_dataset):
    dataset = instantiate_dataset
    dataloader = CDR3DataLoader(
        dataset=dataset,
        batch_size=5,
        batch_optimisation=False
    )
    assert(dataloader.batch_optimisation == False)
    
    for i in range(2):
        for src_batch, tgt_batch in dataloader:
            assert(type(src_batch) == type(tgt_batch) == torch.Tensor)
            assert(src_batch.size() == tgt_batch.size())
            assert(src_batch.size(0) in (4, 5))
            assert(src_batch.size(1) >= 10 and src_batch.size(1) <= 20)
            assert(src_batch.dim() == 2)


def test_dataloader_with_optim(instantiate_dataloader_with_optim):
    dataloader = instantiate_dataloader_with_optim
    assert(dataloader.batch_optimisation == True)

    for i in range(10):
        min_batch_seq_len_encountered = 999
        for src_batch, tgt_batch in dataloader:
            assert(type(src_batch) == type(tgt_batch) == torch.Tensor)
            assert(src_batch.size() == tgt_batch.size())
            assert(src_batch.size(0) in (4, 5))
            assert(src_batch.size(1) >= 10 and src_batch.size(1) <= 20)
            assert(src_batch.dim() == 2)
            min_batch_seq_len_encountered = min(
                src_batch.size(1),
                min_batch_seq_len_encountered
            )
        assert(min_batch_seq_len_encountered == 12)


def test_get_set_jumble(instantiate_dataloader_with_optim):
    dataloader = instantiate_dataloader_with_optim

    dataloader.jumble = True
    assert(dataloader.jumble == True)
    dataloader.jumble = False
    assert(dataloader.jumble == False)


# Negative tests
def test_incorrect_dataset_type():
    dataset = torch.utils.data.Dataset()
    with pytest.raises(AssertionError):
        dataloader = CDR3DataLoader(dataset, 5)


def test_bad_jumble_value(instantiate_dataloader_with_optim):
    dataloader = instantiate_dataloader_with_optim
    with pytest.raises(AssertionError):
        dataloader.jumble = 'True'