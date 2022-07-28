import pandas as pd
import pytest
import random
from source.data_handling.datasets import Cdr3FineTuneDataset


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

        ref_epitope = df.iloc[idx, 0]
        cdr3a, cdr3b, label = ds._get_matched_cdr3(ref_epitope)

        assert label
        assert ref_epitope in \
            df[
                (df['Alpha CDR3'] == cdr3a) &
                (df['Beta CDR3'] == cdr3b)
            ]['Epitope'].unique()


def test_get_unmatched_cdr3(get_dataframe, instantiate_dataset):
    df = get_dataframe
    ds = instantiate_dataset

    for i in range(10):
        idx = random.randrange(len(df))

        ref_epitope = df.iloc[idx, 0]
        cdr3a, cdr3b, label = ds._get_unmatched_cdr3(ref_epitope)

        assert not label
        assert ref_epitope not in \
            df[
                (df['Alpha CDR3'] == cdr3a) &
                (df['Beta CDR3'] == cdr3b)
            ]['Epitope'].unique()


def test_getitem(get_dataframe, instantiate_dataset):
    df = get_dataframe
    ds = instantiate_dataset

    for i in range(10):
        idx = random.randrange(len(df))

        cdr3_1a, cdr3_1b, cdr3_2a, cdr3_2b, label = ds[idx]

        epitopes_1 = set(
            df[df['Alpha CDR3'] == cdr3_1a]['Epitope'].unique()
        )
        epitopes_2 = set(
            df[df['Alpha CDR3'] == cdr3_2a]['Epitope'].unique()
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