import pytest
import random
from src.datahandling import datasets
from src.datahandling.tokenisers import _Tokeniser


@pytest.fixture
def dummy_tokeniser():
    class DummyTokeniser(_Tokeniser):
        '''
        Dummy tokeniser that just returns the row index of an item in the
        dataset as the 'tokenisation'
        '''
        @property
        def vocab_size(self) -> int:
            return 0

        def tokenise(self, tcr, noising: bool = False):
            return (tcr.name, noising)
    
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


    def test_getitem(self, tcr_dataset):
        random_index = random.randrange(0, len(tcr_dataset))

        assert tcr_dataset[random_index] == (random_index, False)


class TestAutoContrastiveDataset:
    def test_getitem(self, mock_data_df, dummy_tokeniser):
        dataset = datasets.AutoContrastiveDataset(
            data=mock_data_df,
            tokeniser=dummy_tokeniser
        )

        random_index = random.randrange(0, len(dataset))

        x, x_prime = dataset[random_index]
        assert x == (random_index, False)
        assert x_prime == (random_index, False)
    

    def test_noising(self, mock_data_df, dummy_tokeniser):
        dataset = datasets.AutoContrastiveDataset(
            data=mock_data_df,
            tokeniser=dummy_tokeniser,
            noising=True
        )

        random_index = random.randrange(0, len(dataset))

        x, x_prime = dataset[random_index]
        assert x == (random_index, False)
        assert x_prime == (random_index, True)


class TestEpitopeContrastiveDataset:
    def test_len(self, mock_data_df, dummy_tokeniser):
        dataset = datasets.EpitopeContrastiveDataset(
            data=mock_data_df,
            tokeniser=dummy_tokeniser
        )
        assert len(dataset) == 3


    def test_getitem(self, mock_data_df, dummy_tokeniser):
        dataset = datasets.EpitopeContrastiveDataset(
            data=mock_data_df,
            tokeniser=dummy_tokeniser
        )

        x, x_prime = dataset[0]
        assert x == (0, False)
        assert x_prime[0] in (0, 1)
        assert x_prime[1] == False

        x, x_prime = dataset[1]
        assert x == (1, False)
        assert x_prime[0] in (0, 1)
        assert x_prime[1] == False

        x, x_prime = dataset[2]
        assert x == (2, False)
        assert x_prime[0] == 2
        assert x_prime[1] == False