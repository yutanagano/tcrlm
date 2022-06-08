from collections import Counter
import os
import random
import re
import pandas as pd
import pytest
from source.data_handling import Cdr3PretrainDataset


@pytest.fixture(scope='module')
def get_path_to_mock_csv(get_path_to_project):
    return os.path.join(
        get_path_to_project,
        'tests/data/mock_unlabelled_data.csv'
    )


@pytest.fixture(scope='module')
def get_dataframe(get_path_to_mock_csv):
    df = pd.read_csv(get_path_to_mock_csv)
    yield df


@pytest.fixture(scope='module')
def instantiate_dataset(get_path_to_mock_csv):
    dataset = Cdr3PretrainDataset(data=get_path_to_mock_csv)
    yield dataset


def reconstruct(x, y):
    '''
    Using the x and y outputs of the pretrain dataset, reconstruct the original
    CDR3 sequence before masking.
    '''
    unmasked_in_y = [p for p in enumerate(y) if p[1] != '-']
    reconstructed = list(x)
    for idx, a_a in unmasked_in_y:
        assert(a_a in 'ACDEFGHIKLMNPQRSTVWY')
        reconstructed[idx] = a_a
    return ''.join(reconstructed)


# Positive tests
def test_loads_csv(instantiate_dataset,get_dataframe):
    dataset = instantiate_dataset
    dataframe = get_dataframe

    assert(dataset._dataframe.equals(dataframe))


def test_loads_csv_directly(get_dataframe):
    dataframe = get_dataframe
    dataset = Cdr3PretrainDataset(
        data=dataframe
    )

    assert(dataset._dataframe.equals(dataframe))

def test_length(instantiate_dataset,get_dataframe):
    dataset = instantiate_dataset
    dataframe = get_dataframe

    assert(len(dataset) == len(dataframe))


def test_get_set_jumble(instantiate_dataset):
    dataset = instantiate_dataset
    dataset.jumble = True
    assert(dataset.jumble == True)
    dataset.jumble = False
    assert(dataset.jumble == False)


def test_getitem(instantiate_dataset, get_dataframe):
    dataset = instantiate_dataset
    dataset.jumble = False
    dataframe = get_dataframe

    random.seed(42)

    for i in range(len(dataframe)):
        cdr3 = dataframe['CDR3'].iloc[i]
        x, y = dataset[i]

        assert(type(x) == type(y) == list)

        x = ''.join(x)
        y = ''.join(y)

        # Ensure x and y are the same length
        assert(len(x) == len(y))

        # Ensure that there are 2 or 3 unmaked residues in y
        unmasked_in_y = [p for p in enumerate(y) if p[1] != '-']
        assert(len(unmasked_in_y) in (2,3))

        # Ensure that combining the masked token from y wth the rest of x is a
        # reconstruction of the original cdr3
        assert(reconstruct(x, y) == cdr3)


def test_jumble_mode(instantiate_dataset, get_dataframe):
    dataset = instantiate_dataset
    dataset.jumble = True
    dataframe = get_dataframe

    random.seed(42)

    for i in range(len(dataframe)):
        cdr3 = dataframe['CDR3'].iloc[i]
        x, y = dataset[i]

        assert(type(x) == type(y) == list)

        x = ''.join(x)
        y = ''.join(y)

        # Ensure x, y, and the known original cdr3 are the same length
        assert(len(x) == len(y) == len(cdr3))

        # Ensure that there are 2 or 3 unmaked residues in y
        unmasked_in_y = [p for p in enumerate(y) if p[1] != '-']
        assert(len(unmasked_in_y) in (2,3))

        # Ensure that combining the masked token from y wth the rest of x is an
        # anagram of the original cdr3 (but not identical)
        with pytest.raises(AssertionError):
            assert(reconstruct(x, y) == cdr3)
        
        assert(sorted(reconstruct(x, y)) == sorted(cdr3))


def test_respect_frequencies(instantiate_dataset, get_dataframe):
    dataset = instantiate_dataset
    dataset.jumble = False
    dataframe = get_dataframe
    
    # Enable respect_frequencies
    dataset.respect_frequencies = True

    # Test getter
    assert(dataset.respect_frequencies == True)
    
    # Test length
    assert(len(dataset) == dataframe['frequency'].sum())

    # Test indexing limits
    assert(reconstruct(*dataset[len(dataset) - 1]) == \
        reconstruct(*dataset[-1]) == \
        dataframe['CDR3'].iloc[-1])
    
    assert(reconstruct(*dataset[0]) == \
        reconstruct(*dataset[-len(dataset)]) == \
        dataframe['CDR3'].iloc[0])

    with pytest.raises(IndexError):
        dataset[len(dataset)]
    
    with pytest.raises(IndexError):
        dataset[-len(dataset) - 1]

    # Test the number of times CDR3s are seen in one full loop through dataset
    cdr3_counter = Counter()

    for x, y in dataset:
        cdr3_counter[reconstruct(x, y)] += 1

    for idx, row in dataframe.iterrows():
        assert(cdr3_counter[row['CDR3']] == row['frequency'])


def test_no_masking(get_dataframe):
    dataframe = get_dataframe
    dataset = Cdr3PretrainDataset(
        data=dataframe,
        p_mask=0
    )

    random.seed(42)

    for i in range(len(dataframe)):
        cdr3 = dataframe['CDR3'].iloc[i]
        x, y = dataset[i]

        assert(type(x) == type(y) == str)

        # Ensure x and y and cdr3 are the same
        assert(x == y == cdr3)


# Negative tests
def test_bad_csv_path():
    with pytest.raises(RuntimeError):
        Cdr3PretrainDataset('/some/bad/path')


def test_nonexistent_csv_path(get_path_to_project):
    with pytest.raises(RuntimeError):
        Cdr3PretrainDataset(os.path.join(get_path_to_project, 'README.md'))


def test_bad_jumble_set(instantiate_dataset):
    dataset = instantiate_dataset
    with pytest.raises(AssertionError):
        dataset.jumble = 'True'


def test_bad_csv_format(get_path_to_project):
    with pytest.raises(RuntimeError):
        Cdr3PretrainDataset(
            os.path.join(get_path_to_project, 'tests/data/bad_format.csv')
        )