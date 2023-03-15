import pandas as pd
import pytest
from src.datahandling import tokenisers
import torch


@pytest.fixture
def mock_data_df():
    df = pd.read_csv("tests/resources/mock_data.csv", dtype="string")
    return df


@pytest.fixture
def mock_data_beta_df():
    df = pd.read_csv("tests/resources/mock_data_beta.csv", dtype="string")
    return df


class TestCDR3Tokeniser:
    def test_tokenise(self, mock_data_df):
        tokeniser = tokenisers.CDR3Tokeniser()

        expected = [
            torch.tensor(
                [
                    [2, 0, 0, 0],
                    [4, 1, 6, 1],
                    [3, 2, 6, 1],
                    [18, 3, 6, 1],
                    [16, 4, 6, 1],
                    [22, 5, 6, 1],
                    [7, 6, 6, 1],
                    [4, 1, 5, 2],
                    [3, 2, 5, 2],
                    [19, 3, 5, 2],
                    [22, 4, 5, 2],
                    [21, 5, 5, 2],
                ]
            ),
            torch.tensor(
                [
                    [2, 0, 0, 0],
                    [4, 1, 6, 1],
                    [3, 2, 6, 1],
                    [18, 3, 6, 1],
                    [16, 4, 6, 1],
                    [22, 5, 6, 1],
                    [7, 6, 6, 1],
                ]
            ),
            torch.tensor(
                [
                    [2, 0, 0, 0],
                    [4, 1, 5, 2],
                    [3, 2, 5, 2],
                    [19, 3, 5, 2],
                    [22, 4, 5, 2],
                    [21, 5, 5, 2],
                ]
            ),
        ]

        for (_, item), target in zip(mock_data_df.iterrows(), expected):
            assert torch.equal(tokeniser.tokenise(item), target)

    def test_vocab_size(self):
        tokeniser = tokenisers.CDR3Tokeniser()

        assert tokeniser.vocab_size == 20


class TestBCDR3Tokeniser:
    def test_tokenise(self, mock_data_beta_df):
        tokeniser = tokenisers.BCDR3Tokeniser(p_drop_aa=0)

        expected = [
            torch.tensor(
                [[2, 0, 0], [4, 1, 5], [3, 2, 5], [19, 3, 5], [22, 4, 5], [21, 5, 5]]
            ),
            torch.tensor(
                [[2, 0, 0], [4, 1, 5], [3, 2, 5], [16, 3, 5], [22, 4, 5], [7, 5, 5]]
            ),
        ]

        for (_, item), target in zip(mock_data_beta_df.iterrows(), expected):
            assert torch.equal(tokeniser.tokenise(item), target)

    def test_vocab_size(self):
        tokeniser = tokenisers.BCDR3Tokeniser(p_drop_aa=0)

        assert tokeniser.vocab_size == 20


class TestBVCDR3Tokeniser:
    def test_tokenise(self, mock_data_beta_df):
        tokeniser = tokenisers.BVCDR3Tokeniser()

        expected = [
            torch.tensor(
                [
                    [2, 0, 0, 0],
                    [38, 0, 0, 1],
                    [4, 1, 5, 2],
                    [3, 2, 5, 2],
                    [19, 3, 5, 2],
                    [22, 4, 5, 2],
                    [21, 5, 5, 2],
                ]
            ),
            torch.tensor(
                [
                    [2, 0, 0, 0],
                    [46, 0, 0, 1],
                    [4, 1, 5, 2],
                    [3, 2, 5, 2],
                    [16, 3, 5, 2],
                    [22, 4, 5, 2],
                    [7, 5, 5, 2],
                ]
            ),
        ]

        for (_, item), target in zip(mock_data_beta_df.iterrows(), expected):
            assert torch.equal(tokeniser.tokenise(item), target)

    def test_vocab_size(self):
        tokeniser = tokenisers.BVCDR3Tokeniser()

        assert tokeniser.vocab_size == 68


class TestBCDRTokeniser:
    def test_tokenise(self, mock_data_beta_df):
        tokeniser = tokenisers.BCDRTokeniser(p_drop_aa=0, p_drop_cdr=0)

        expected = [
            torch.tensor(
                [
                    [2, 0, 0, 0],
                    [18, 1, 5, 1],
                    [14, 2, 5, 1],
                    [9, 3, 5, 1],
                    [12, 4, 5, 1],
                    [22, 5, 5, 1],  # SNHLY
                    [7, 1, 6, 2],
                    [22, 2, 6, 2],
                    [14, 3, 6, 2],
                    [14, 4, 6, 2],
                    [6, 5, 6, 2],
                    [10, 6, 6, 2],  # FYNNEI
                    [4, 1, 5, 3],
                    [3, 2, 5, 3],
                    [19, 3, 5, 3],
                    [22, 4, 5, 3],
                    [21, 5, 5, 3],
                ]
            ),
            torch.tensor(
                [
                    [2, 0, 0, 0],
                    [8, 1, 6, 1],
                    [19, 2, 6, 1],
                    [18, 3, 6, 1],
                    [14, 4, 6, 1],
                    [15, 5, 6, 1],
                    [14, 6, 6, 1],  # GTSNPN
                    [18, 1, 5, 2],
                    [20, 2, 5, 2],
                    [8, 3, 5, 2],
                    [10, 4, 5, 2],
                    [8, 5, 5, 2],  # SVGIG
                    [4, 1, 5, 3],
                    [3, 2, 5, 3],
                    [16, 3, 5, 3],
                    [22, 4, 5, 3],
                    [7, 5, 5, 3],
                ]
            ),
        ]

        for (_, item), target in zip(mock_data_beta_df.iterrows(), expected):
            assert torch.equal(tokeniser.tokenise(item), target)


class TestCDRTokeniser:
    def test_tokenise(self, mock_data_df):
        tokeniser = tokenisers.CDRTokeniser(p_drop_aa=0, p_drop_cdr=0, p_drop_chain=0)

        expected = [
            torch.tensor(
                [
                    [2, 0, 0, 0],
                    [19, 1, 6, 1],
                    [18, 2, 6, 1],
                    [8, 3, 6, 1],
                    [7, 4, 6, 1],
                    [22, 5, 6, 1],
                    [8, 6, 6, 1],  # TSGFYG
                    [14, 1, 6, 2],
                    [3, 2, 6, 2],
                    [12, 3, 6, 2],
                    [5, 4, 6, 2],
                    [8, 5, 6, 2],
                    [12, 6, 6, 2],  # NALDGL
                    [4, 1, 6, 3],
                    [3, 2, 6, 3],
                    [18, 3, 6, 3],
                    [16, 4, 6, 3],
                    [22, 5, 6, 3],
                    [7, 6, 6, 3],  # CASQYF
                    [18, 1, 5, 4],
                    [14, 2, 5, 4],
                    [9, 3, 5, 4],
                    [12, 4, 5, 4],
                    [22, 5, 5, 4],  # SNHLY
                    [7, 1, 6, 5],
                    [22, 2, 6, 5],
                    [14, 3, 6, 5],
                    [14, 4, 6, 5],
                    [6, 5, 6, 5],
                    [10, 6, 6, 5],  # FYNNEI
                    [4, 1, 5, 6],
                    [3, 2, 5, 6],
                    [19, 3, 5, 6],
                    [22, 4, 5, 6],
                    [21, 5, 5, 6],  # CATYW
                ]
            ),
            torch.tensor(
                [
                    [2, 0, 0, 0],
                    [19, 1, 6, 1],
                    [18, 2, 6, 1],
                    [8, 3, 6, 1],
                    [7, 4, 6, 1],
                    [22, 5, 6, 1],
                    [8, 6, 6, 1],  # TSGFYG
                    [14, 1, 6, 2],
                    [3, 2, 6, 2],
                    [12, 3, 6, 2],
                    [5, 4, 6, 2],
                    [8, 5, 6, 2],
                    [12, 6, 6, 2],  # NALDGL
                    [4, 1, 6, 3],
                    [3, 2, 6, 3],
                    [18, 3, 6, 3],
                    [16, 4, 6, 3],
                    [22, 5, 6, 3],
                    [7, 6, 6, 3],  # CASQYF
                ]
            ),
            torch.tensor(
                [
                    [2, 0, 0, 0],
                    [18, 1, 5, 4],
                    [14, 2, 5, 4],
                    [9, 3, 5, 4],
                    [12, 4, 5, 4],
                    [22, 5, 5, 4],  # SNHLY
                    [7, 1, 6, 5],
                    [22, 2, 6, 5],
                    [14, 3, 6, 5],
                    [14, 4, 6, 5],
                    [6, 5, 6, 5],
                    [10, 6, 6, 5],  # FYNNEI
                    [4, 1, 5, 6],
                    [3, 2, 5, 6],
                    [19, 3, 5, 6],
                    [22, 4, 5, 6],
                    [21, 5, 5, 6],  # CATYW
                ]
            ),
        ]

        for (_, item), target in zip(mock_data_df.iterrows(), expected):
            assert torch.equal(tokeniser.tokenise(item), target)
