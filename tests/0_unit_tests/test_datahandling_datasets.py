import pandas as pd
from pathlib import Path
import pytest
import random
from src.datahandling import datasets
from src.datahandling.tokenisers import Tokeniser


@pytest.fixture
def mock_data_path():
    return Path('tests/resources/mock_data.csv')


@pytest.fixture
def mock_data_df(mock_data_path):
    df = pd.read_csv(mock_data_path, dtype='string')
    return df


@pytest.fixture
def dummy_tokeniser():
    class DummyTokeniser(Tokeniser):
        def tokenise(self, x):
            return x
    
    return DummyTokeniser()


@pytest.fixture
def tcr_dataset(mock_data_df, dummy_tokeniser):
    dataset = datasets.TcrDataset(
        data=mock_data_df,
        tokeniser=dummy_tokeniser
    )

    return dataset


class TestTcrDataset:
    def test_init_dataframe(self, mock_data_df, dummy_tokeniser):
        dataset = datasets.TcrDataset(
            data=mock_data_df,
            tokeniser=dummy_tokeniser
        )

        assert dataset._data.equals(mock_data_df)


    def test_init_path(self, mock_data_path, mock_data_df, dummy_tokeniser):
        dataset = datasets.TcrDataset(
            data=mock_data_path,
            tokeniser=dummy_tokeniser
        )

        assert dataset._data.equals(mock_data_df)


    def test_len(self, tcr_dataset, mock_data_df):
        assert len(tcr_dataset) == len(mock_data_df)


    def test_getitem(self, tcr_dataset, mock_data_df):
        random_index = random.randrange(0, len(tcr_dataset))

        assert tcr_dataset[random_index].equals(mock_data_df.iloc[random_index])