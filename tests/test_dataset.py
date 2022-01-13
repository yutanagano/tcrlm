import os
import random
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


@pytest.fixture(scope='module')
def instantiate_dataset_with_masking(get_path_to_mock_csv):
    dataset = CDR3Dataset(path_to_csv=get_path_to_mock_csv,
                          p_masked=0.1)
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


def test_getitem_nomasking(instantiate_dataset,get_dataframe):
    dataset = instantiate_dataset
    dataframe = get_dataframe

    for i in range(10):
        index = random.randint(0,len(dataframe)-1)
        cdr3 = dataframe['CDR3'].iloc[index]
        
        x, y = dataset[index]

        assert(y == cdr3)
        assert(len(x) == len(y))
        assert(not '?' in x)


def test_getitem_masking(instantiate_dataset_with_masking, get_dataframe):
    dataset = instantiate_dataset_with_masking
    dataframe = get_dataframe

    for i in range(10):
        index = random.randint(0,len(dataframe)-1)
        cdr3 = dataframe['CDR3'].iloc[index]

        x, y = dataset[index]

        assert(y == cdr3)
        assert(len(x) == len(y))
        assert('?' in x)


# Negative tests
def test_bad_csv_path():
    with pytest.raises(RuntimeError):
        CDR3Dataset('/some/bad/path')


def test_nonexistent_csv_path(get_path_to_project):
    with pytest.raises(RuntimeError):
        CDR3Dataset(os.path.join(get_path_to_project, 'README.md'))