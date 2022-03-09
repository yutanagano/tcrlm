import pytest
from source.data_handling import Cdr3PretrainDataset, PadMinimalBatchSampler


@pytest.fixture(scope='module')
def instantiate_dataset():
    dataset = Cdr3PretrainDataset('tests/data/mock_unlabelled_data.csv')
    yield dataset


# Positive tests
def test_iter(instantiate_dataset):
    dataset = instantiate_dataset
    sampler = PadMinimalBatchSampler(
        data_source=dataset,
        batch_size=5
    )
    for i in range(2):
        for batch in sampler:
            assert(len(batch) in (4,5))
    
    sampler.shuffle = True
    for i in range(2):
        for batch in sampler:
            assert(len(batch) in (4,5))


def test_len(instantiate_dataset):
    dataset = instantiate_dataset
    sampler = PadMinimalBatchSampler(
        data_source=dataset,
        batch_size=5
    )
    assert(len(sampler) == 6)


def test_min_batch_seq_len(instantiate_dataset):
    dataset = instantiate_dataset
    sampler = PadMinimalBatchSampler(
        data_source=dataset,
        batch_size=5
    )

    def get_batch_seq_len(batch_indices):
        max_seq_len = 0
        for index in batch_indices:
            l = dataset.get_length(index)
            max_seq_len = max(l, max_seq_len)
        return max_seq_len

    for i in range(10):
        min_batch_seq_len_encountered = 999
        for batch in sampler:
            batch_seq_len = get_batch_seq_len(batch)
            min_batch_seq_len_encountered = min(
                batch_seq_len,
                min_batch_seq_len_encountered
            )
        assert(min_batch_seq_len_encountered == 12)


# Negative tests
def test_bad_data_source():
    data = [
        'asdf',
        'asdf',
        'asdf',
        'asdf',
        'asdf',
        'asdf',
        'asdf',
        'asdf'
    ]
    with pytest.raises(AssertionError):
        sampler = PadMinimalBatchSampler(data, 3)