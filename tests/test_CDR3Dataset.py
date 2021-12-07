import os
import random
from itertools import product
import pandas as pd
import numpy as np
import pytest
from data_handling import SequenceConverter, CDR3Dataset


@pytest.fixture(scope='module')
def get_path_to_mock_csv(get_path_to_project):
    return os.path.join(get_path_to_project, 'tests/data/mock_data.csv')


@pytest.fixture(scope='module')
def get_dataframe(get_path_to_mock_csv):
    df = pd.read_csv(get_path_to_mock_csv)
    yield df


@pytest.fixture(scope='module')
def instantiate_dataset(get_path_to_mock_csv):
    dataset = CDR3Dataset(
            path_to_csv = get_path_to_mock_csv)
    yield dataset


@pytest.fixture(scope='module')
def instantiate_converter_0_padding():
    converter = SequenceConverter(padding=0)
    yield converter


# Positive tests
def test_loads_csv(instantiate_dataset,get_dataframe):
    dataset = instantiate_dataset
    dataframe = get_dataframe

    assert(dataset.dataframe.equals(dataframe))


def test_length(instantiate_dataset,get_dataframe):
    dataset = instantiate_dataset
    dataframe = get_dataframe

    assert(len(dataset) == len(dataframe))


def test_getitem(instantiate_dataset,instantiate_converter_0_padding,get_dataframe):
    dataset = instantiate_dataset
    converter = instantiate_converter_0_padding
    dataframe = get_dataframe

    for i in range(10):
        index = random.randint(0,len(dataframe)-1)
        cdr3 = dataframe['CDR3'].iloc[index]
        atchley_encoding = converter.to_atchley(cdr3)
        one_hot_encoding = converter.to_one_hot(cdr3)
        
        x, y = dataset[index]

        assert(np.array_equal(x,atchley_encoding))
        assert(np.array_equal(y,one_hot_encoding))


def test_x_y_form_permutations(instantiate_converter_0_padding,get_dataframe,get_path_to_mock_csv):
    converter = instantiate_converter_0_padding
    dataframe = get_dataframe

    cdr3 = dataframe['CDR3'].iloc[0]
    atchley = converter.to_atchley(cdr3)
    one_hot = converter.to_one_hot(cdr3)

    for x_atchley, y_atchley in product((True, False), repeat=2):
        dataset = CDR3Dataset(get_path_to_mock_csv,x_atchley=x_atchley,y_atchley=y_atchley)
        x, y = dataset[0]
        
        if x_atchley: assert(np.array_equal(x, atchley))
        else: assert(np.array_equal(x, one_hot))

        if y_atchley: assert(np.array_equal(y, atchley))
        else: assert(np.array_equal(y, one_hot))


# Negative tests
def test_bad_csv_path():
    with pytest.raises(RuntimeError):
        CDR3Dataset('/some/bad/path')


def test_nonexistent_csv_path(get_path_to_project):
    with pytest.raises(RuntimeError):
        CDR3Dataset(os.path.join(get_path_to_project, 'README.md'))