from itertools import product
import pytest
import random
from src.datahandling import datasets
from src.datahandling.tokenisers import _Tokeniser


@pytest.fixture
def dummy_tokeniser():
    class DummyTokeniser(_Tokeniser):
        """
        Dummy tokeniser that just returns the row index of an item in the
        dataset as the 'tokenisation'
        """

        @property
        def vocab_size(self) -> int:
            return 0

        def tokenise(self, tcr, noising: bool = False):
            return (tcr.name, noising)

    return DummyTokeniser()


@pytest.fixture
def tcr_dataset(mock_data_df, dummy_tokeniser):
    dataset = datasets.TCRDataset(data=mock_data_df, tokeniser=dummy_tokeniser)

    return dataset


class TestTcrDataset:
    def test_init_dataframe(self, mock_data_df, dummy_tokeniser):
        dataset = datasets.TCRDataset(data=mock_data_df, tokeniser=dummy_tokeniser)

        assert dataset._data.equals(mock_data_df)

    def test_init_path(self, mock_data_path, mock_data_df, dummy_tokeniser):
        dataset = datasets.TCRDataset(data=mock_data_path, tokeniser=dummy_tokeniser)

        assert dataset._data.equals(mock_data_df)

    def test_len(self, tcr_dataset, mock_data_df):
        assert len(tcr_dataset) == len(mock_data_df)

    def test_getitem(self, tcr_dataset):
        random_index = random.randrange(0, len(tcr_dataset))

        assert tcr_dataset[random_index] == (random_index, False)


class TestAutoContrastiveDataset:
    @pytest.mark.parametrize(
        ("noising_lhs", "noising_rhs"), product((True, False), repeat=2)
    )
    def test_getitem(self, mock_data_df, dummy_tokeniser, noising_lhs, noising_rhs):
        dataset = datasets.AutoContrastiveDataset(
            data=mock_data_df,
            tokeniser=dummy_tokeniser,
            censoring_lhs=noising_lhs,
            censoring_rhs=noising_rhs,
        )

        random_index = random.randrange(0, len(dataset))

        x, x_lhs, x_rhs = dataset[random_index]
        assert x == (random_index, False)
        assert x_lhs == (random_index, noising_lhs)
        assert x_rhs == (random_index, noising_rhs)


class TestEpitopeContrastiveDataset_dep:
    def test_len(self, mock_data_df, dummy_tokeniser):
        dataset = datasets.EpitopeContrastiveDataset_dep(
            data=mock_data_df,
            tokeniser=dummy_tokeniser,
            censoring_lhs=False,
            censoring_rhs=False,
        )
        assert len(dataset) == 2

    def test_getitem(self, mock_data_df, dummy_tokeniser):
        dataset = datasets.EpitopeContrastiveDataset_dep(
            data=mock_data_df,
            tokeniser=dummy_tokeniser,
            censoring_lhs=False,
            censoring_rhs=False,
        )

        sample = dataset[0]
        assert len(sample) == 2
        assert sample[0][0] == (0, False)
        assert sample[1][0] == (2, False)

        sample = dataset[1]
        assert len(sample) == 2
        assert sample[0][0] == (1, False)
        assert sample[1][0] == (2, False)

    def test_internal_shuffle(self, mock_data_df, dummy_tokeniser):
        dataset = datasets.EpitopeContrastiveDataset_dep(
            data=mock_data_df,
            tokeniser=dummy_tokeniser,
            censoring_lhs=False,
            censoring_rhs=False,
        )

        dataset._internal_shuffle(0)

        sample = dataset[0]
        assert sample[0][0] == (1, False)
        assert sample[1][0] == (2, False)


class TestEpitopeContrastiveDataset:
    def test_len(self, mock_data_large_df, dummy_tokeniser):
        dataset = datasets.EpitopeContrastiveDataset(
            data=mock_data_large_df,
            tokeniser=dummy_tokeniser,
            censoring_lhs=False,
            censoring_rhs=False,
        )
        assert len(dataset) == 6 + 3

    def test_getitem(self, mock_data_large_df, dummy_tokeniser):
        dataset = datasets.EpitopeContrastiveDataset(
            data=mock_data_large_df,
            tokeniser=dummy_tokeniser,
            censoring_lhs=False,
            censoring_rhs=False,
        )

        x, x_lhs, x_rhs = dataset[0]
        assert x == x_lhs == (0, False)
        assert x_rhs == (1, False)

        x, x_lhs, x_rhs = dataset[2]
        assert x == x_lhs == (0, False)
        assert x_rhs == (3, False)

        x, x_lhs, x_rhs = dataset[4]
        assert x == x_lhs == (1, False)
        assert x_rhs == (3, False)

        x, x_lhs, x_rhs = dataset[6]
        assert x == x_lhs == (4, False)
        assert x_rhs == (5, False)

        x, x_lhs, x_rhs = dataset[8]
        assert x == x_lhs == (5, False)
        assert x_rhs == (6, False)

        x, x_lhs, x_rhs = dataset[-1]
        assert x == x_lhs == (5, False)
        assert x_rhs == (6, False)

        x, x_lhs, x_rhs = dataset[-9]
        assert x == x_lhs == (0, False)
        assert x_rhs == (1, False)
    
    def test_out_of_range(self, mock_data_large_df, dummy_tokeniser):
        dataset = datasets.EpitopeContrastiveDataset(
            data=mock_data_large_df,
            tokeniser=dummy_tokeniser,
            censoring_lhs=False,
            censoring_rhs=False,
        )

        with pytest.raises(IndexError):
            dataset[9]
        
        with pytest.raises(IndexError):
            dataset[-10]
    
    def test_iterate(self, mock_data_large_df, dummy_tokeniser):
        dataset = datasets.EpitopeContrastiveDataset(
            data=mock_data_large_df,
            tokeniser=dummy_tokeniser,
            censoring_lhs=False,
            censoring_rhs=False,
        )

        expected_length = 9
        expected_items = [
            (0,1),
            (0,2),
            (0,3),
            (1,2),
            (1,3),
            (2,3),
            (4,5),
            (4,6),
            (5,6)
        ]

        for idx, (x, x_lhs, x_rhs) in enumerate(dataset):
            assert (x_lhs[0], x_rhs[0]) == expected_items[idx]

        assert idx == expected_length - 1