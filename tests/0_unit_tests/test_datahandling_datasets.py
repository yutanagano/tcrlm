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
        @property
        def vocab_size(self) -> int:
            return 0

        def tokenise(self, tcr, chain: str = 'both'):
            return (tcr.name, chain)
    
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

        assert tcr_dataset[random_index] == (random_index, 'both')


class TestUnsupervisedSimCLDataset:
    def test_getitem(self, mock_data_df, dummy_tokeniser):
        dataset = datasets.UnsupervisedSimCLDataset(
            data=mock_data_df,
            tokeniser=dummy_tokeniser
        )

        random_index = random.randrange(0, len(dataset))

        x, x_prime = dataset[random_index]
        assert x == (random_index, 'both')
        assert x_prime[0] == random_index
        assert x_prime[1] in ('both', 'alpha', 'beta')

class TestSupervisedSimCLDataset:
    def test_len(self, mock_data_df, dummy_tokeniser):
        dataset = datasets.SupervisedSimCLDataset(
            data=mock_data_df,
            tokeniser=dummy_tokeniser
        )
        assert len(dataset) == 4


    def test_getitem(self, mock_data_df, dummy_tokeniser):
        dataset = datasets.SupervisedSimCLDataset(
            data=mock_data_df,
            tokeniser=dummy_tokeniser
        )

        x, x_prime = dataset[0]
        assert x == (0, 'both')
        assert x_prime[0] in (0, 1)
        assert x_prime[1] == 'both'

        x, x_prime = dataset[1]
        assert x == (2, 'both')
        assert x_prime[0] == 2
        assert x_prime[1] == 'both'

        x, x_prime = dataset[2]
        assert x == (1, 'both')
        assert x_prime[0] in (0, 1)
        assert x_prime[1] == 'both'

        x, x_prime = dataset[3]
        assert x == (2, 'both')
        assert x_prime[0] == 2
        assert x_prime[1] == 'both'