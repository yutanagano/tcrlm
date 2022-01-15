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

    assert(dataset.dataframe.equals(dataframe))


def test_length(instantiate_dataset,get_dataframe):
    dataset = instantiate_dataset
    dataframe = get_dataframe

    assert(len(dataset) == len(dataframe))


def test_getitem(instantiate_dataset, get_dataframe):
    dataset = instantiate_dataset
    dataframe = get_dataframe

    for i in range(len(dataframe)):
        cdr3 = dataframe['CDR3'].iloc[i]
        x, y = dataset[i]

        # Ensure x and y are the same length
        assert(len(x) == len(y))

        # Ensure that there is an unmaked residue in y
        match = re.search(r'[^-]',y)
        assert(match)

        # Ensure there is only one match
        assert(len(re.findall(r'[^-]',y)) == 1)

        # Ensure that that unmaked residue is a valid token/amino acid
        a_idx = match.start()
        a_a = match[0]
        assert(a_a in 'ACDEFGHIKLMNPQRSTVWY')

        # Ensure that combining the masked token from y wth the rest of x is a
        # reconstruction of the original cdr3
        reconstructed = list(x)
        reconstructed[a_idx] = a_a
        assert(''.join(reconstructed) == cdr3)


# Negative tests
def test_bad_csv_path():
    with pytest.raises(RuntimeError):
        CDR3Dataset('/some/bad/path')


def test_nonexistent_csv_path(get_path_to_project):
    with pytest.raises(RuntimeError):
        CDR3Dataset(os.path.join(get_path_to_project, 'README.md'))