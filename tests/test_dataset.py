import os
import random
import re
import pandas as pd
import pytest
from source.data_handling import CDR3Dataset


@pytest.fixture(scope='module')
def get_path_to_mock_csv(get_path_to_project):
    return os.path.join(get_path_to_project, 'tests/data/mock_data.csv')


@pytest.fixture(scope='module')
def get_dataframe(get_path_to_mock_csv):
    df = pd.read_csv(get_path_to_mock_csv)
    yield df


@pytest.fixture(scope='module')
def instantiate_dataset(get_path_to_mock_csv):
    dataset = CDR3Dataset(path_to_csv=get_path_to_mock_csv)
    yield dataset


# Positive tests
def test_loads_csv(instantiate_dataset,get_dataframe):
    dataset = instantiate_dataset
    dataframe = get_dataframe

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

        # Ensure that that unmaked residue is a valid token/amino acid, and also
        # attempt constructing the original amino acid sequence by combining the
        # x and y sequences
        reconstructed = list(x)
        for idx, a_a in unmasked_in_y:
            assert(a_a in 'ACDEFGHIKLMNPQRSTVWY')
            reconstructed[idx] = a_a

        # Ensure that combining the masked token from y wth the rest of x is a
        # reconstruction of the original cdr3
        assert(''.join(reconstructed) == cdr3)


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

        # Ensure that that unmaked residue is a valid token/amino acid, and also
        # attempt constructing the original amino acid sequence by combining the
        # x and y sequences
        reconstructed = list(x)
        for idx, a_a in unmasked_in_y:
            assert(a_a in 'ACDEFGHIKLMNPQRSTVWY')
            reconstructed[idx] = a_a

        # Ensure that combining the masked token from y wth the rest of x is an
        # anagram of the original cdr3 (but not identical)
        with pytest.raises(AssertionError):
            assert(''.join(reconstructed) == cdr3)
        
        assert(sorted(reconstructed) == sorted(cdr3))


# Negative tests
def test_bad_csv_path():
    with pytest.raises(RuntimeError):
        CDR3Dataset('/some/bad/path')


def test_nonexistent_csv_path(get_path_to_project):
    with pytest.raises(RuntimeError):
        CDR3Dataset(os.path.join(get_path_to_project, 'README.md'))


def test_bad_jumble_set(instantiate_dataset):
    dataset = instantiate_dataset
    with pytest.raises(AssertionError):
        dataset.jumble = 'True'