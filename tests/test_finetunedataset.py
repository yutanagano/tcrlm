import pandas as pd
import pytest
import random
from source.data_handling import Cdr3FineTuneDataset


@pytest.fixture(scope='module')
def get_path_to_mock_data():
    return 'tests/data/mock_labelled_data.csv'


@pytest.fixture(scope='module')
def get_dataframe(get_path_to_mock_data):
    df = pd.read_csv(get_path_to_mock_data)
    yield df


@pytest.fixture(scope='module')
def instantiate_dataset(get_path_to_mock_data):
    dataset = Cdr3FineTuneDataset(get_path_to_mock_data)
    yield dataset


# Positive tests
def test_len(get_dataframe, instantiate_dataset):
    df = get_dataframe
    ds = instantiate_dataset

    assert(len(df) == len(ds))


def test_get_matched_cdr3(get_dataframe, instantiate_dataset):
    df = get_dataframe
    ds = instantiate_dataset

    for i in range(10):
        idx = random.randrange(len(df))

        epitope_1, cdr3_1 = df.iloc[idx, 0:2]
        epitope_2, cdr3_2, label = ds._get_matched_cdr3(idx)

        assert(epitope_1 == epitope_2)
        assert(label)
        assert(epitope_1 in df[df['CDR3'] == cdr3_2]['Epitope'].unique())


def test_get_unmatched_cdr3(get_dataframe, instantiate_dataset):
    df = get_dataframe
    ds = instantiate_dataset

    for i in range(10):
        idx = random.randrange(len(df))

        epitope_1, cdr3_1 = df.iloc[idx, 0:2]
        epitope_2, cdr3_2, label = ds._get_unmatched_cdr3(idx)

        assert(epitope_1 != epitope_2)
        assert(not label)
        assert(epitope_2 in df[df['CDR3'] == cdr3_2]['Epitope'].unique())


def test_getitem(get_dataframe, instantiate_dataset):
    df = get_dataframe
    ds = instantiate_dataset

    for i in range(10):
        idx = random.randrange(len(df))

        cdr3_1, cdr3_2, label = ds[idx]

        epitopes_1 = set(
            df[df['CDR3'] == cdr3_1]['Epitope'].unique()
        )
        epitopes_2 = set(
            df[df['CDR3'] == cdr3_2]['Epitope'].unique()
        )

        if label:
            assert(not epitopes_1.isdisjoint(epitopes_2))
        else:
            assert(epitopes_1.isdisjoint(epitopes_2))


# Negative tests
def test_bad_p_matched_pair(get_path_to_mock_data):
    with pytest.raises(RuntimeError):
        ds = Cdr3FineTuneDataset(get_path_to_mock_data, 0)


def test_bad_csv_path():
    with pytest.raises(RuntimeError):
        ds = Cdr3FineTuneDataset('tests/data/thisdoesntexist.csv')


def test_bad_csv_format():
    with pytest.raises(RuntimeError):
        ds = Cdr3FineTuneDataset('tests/data/bad_format.csv')