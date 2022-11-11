import pytest
import random
from src.datahandling import datasets
from src.datahandling.tokenisers import Tokeniser


@pytest.fixture
def dummy_tokeniser():
    class DummyTokeniser(Tokeniser):
        '''
        Dummy tokeniser that just returns the row index of an item in the
        dataset as the 'tokenisation'
        '''
        def tokenise(self, x):
            return x.name
    
    return DummyTokeniser()


@pytest.fixture
def tcr_dataset(mock_data_df, dummy_tokeniser):
    dataset = datasets.TCRDataset(
        data=mock_data_df,
        tokeniser=dummy_tokeniser
    )

    return dataset


class TestTcrDataset:
    def test_init_dataframe(self, mock_data_df, dummy_tokeniser):
        dataset = datasets.TCRDataset(
            data=mock_data_df,
            tokeniser=dummy_tokeniser
        )

        assert dataset._data.equals(mock_data_df)


    def test_init_path(self, mock_data_path, mock_data_df, dummy_tokeniser):
        dataset = datasets.TCRDataset(
            data=mock_data_path,
            tokeniser=dummy_tokeniser
        )

        assert dataset._data.equals(mock_data_df)


    def test_len(self, tcr_dataset, mock_data_df):
        assert len(tcr_dataset) == len(mock_data_df)


    def test_getitem(self, tcr_dataset, mock_data_df):
        random_index = random.randrange(0, len(tcr_dataset))

        assert tcr_dataset[random_index] == random_index