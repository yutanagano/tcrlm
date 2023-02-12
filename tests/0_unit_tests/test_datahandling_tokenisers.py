import pandas as pd
import pytest
from src.datahandling import tokenisers
import torch


@pytest.fixture
def mock_data_df():
    df = pd.read_csv('tests/resources/mock_data.csv', dtype='string')
    return df


@pytest.fixture
def mock_data_beta_df():
    df = pd.read_csv('tests/resources/mock_data_beta.csv', dtype='string')
    return df


class TestABCDR3Tokeniser:
    def test_tokenise(self, mock_data_df):
        tokeniser = tokenisers.ABCDR3Tokeniser()

        expected = [
            torch.tensor([
                [2,0,0],
                [4,1,1],[3,2,1],[18,3,1],[16,4,1],[22,5,1],[7,6,1],
                [4,1,2],[3,2,2],[19,3,2],[22,4,2],[21,5,2]
            ]),
            torch.tensor([
                [2,0,0],
                [4,1,1],[3,2,1],[18,3,1],[16,4,1],[22,5,1],[7,6,1]
            ]),
            torch.tensor([
                [2,0,0],
                [4,1,2],[3,2,2],[19,3,2],[22,4,2],[21,5,2]
            ]),
        ]

        for (_, item), target in zip(mock_data_df.iterrows(), expected):
            assert torch.equal(tokeniser.tokenise(item), target)


    @pytest.mark.parametrize(
        ('chain', 'expected'),
        (
            (
                'alpha',
                [
                    torch.tensor([
                        [2,0,0],
                        [4,1,1],[3,2,1],[18,3,1],[16,4,1],[22,5,1],[7,6,1]
                    ]),
                    torch.tensor([
                        [2,0,0],
                        [4,1,1],[3,2,1],[18,3,1],[16,4,1],[22,5,1],[7,6,1]
                    ]),
                    torch.tensor([
                        [2,0,0],
                        [4,1,2],[3,2,2],[19,3,2],[22,4,2],[21,5,2]
                    ]),
                ]
            ),
            (
                'beta',
                [
                    torch.tensor([
                        [2,0,0],
                        [4,1,2],[3,2,2],[19,3,2],[22,4,2],[21,5,2]
                    ]),
                    torch.tensor([
                        [2,0,0],
                        [4,1,1],[3,2,1],[18,3,1],[16,4,1],[22,5,1],[7,6,1]
                    ]),
                    torch.tensor([
                        [2,0,0],
                        [4,1,2],[3,2,2],[19,3,2],[22,4,2],[21,5,2]
                    ]),
                ]
            ),
        )
    )
    def test_chain(self, mock_data_df, chain, expected):
        tokeniser = tokenisers.ABCDR3Tokeniser()

        for (_, item), target in zip(mock_data_df.iterrows(), expected):
            assert torch.equal(tokeniser.tokenise(item, chain), target)


    def test_vocab_size(self):
        tokeniser = tokenisers.ABCDR3Tokeniser()

        assert tokeniser.vocab_size == 20


class TestBCDR3Tokeniser:
    def test_tokenise(self, mock_data_beta_df):
        tokeniser = tokenisers.BCDR3Tokeniser()

        expected = [
            torch.tensor([
                [2,0],
                [4,1],[3,2],[19,3],[22,4],[21,5]
            ]),
            torch.tensor([
                [2,0],
                [4,1],[3,2],[16,3],[22,4],[7,5]
            ]),
        ]

        for (_, item), target in zip(mock_data_beta_df.iterrows(), expected):
            assert torch.equal(tokeniser.tokenise(item), target)


    def test_vocab_size(self):
        tokeniser = tokenisers.ABCDR3Tokeniser()

        assert tokeniser.vocab_size == 20


class TestBVCDR3Tokeniser:
    def test_tokenise(self, mock_data_beta_df):
        tokeniser = tokenisers.BVCDR3Tokeniser()

        expected = [
            torch.tensor([
                [2,0,0],
                [38,0,1],
                [4,1,2],[3,2,2],[19,3,2],[22,4,2],[21,5,2]
            ]),
            torch.tensor([
                [2,0,0],
                [46,0,1],
                [4,1,2],[3,2,2],[16,3,2],[22,4,2],[7,5,2]
            ])
        ]

        for (_, item), target in zip(mock_data_beta_df.iterrows(), expected):
            assert torch.equal(tokeniser.tokenise(item), target)


    def test_vocab_size(self):
        tokeniser = tokenisers.BVCDR3Tokeniser()

        assert tokeniser.vocab_size == 68


class TestBCDR123Tokeniser:
    def test_tokenise(self, mock_data_beta_df):
        tokeniser = tokenisers.BCDR123Tokeniser()

        expected = [
            torch.tensor([
                [2,0,0],
                [18,1,1],[14,2,1],[9,3,1],[12,4,1],[22,5,1], #SNHLY
                [7,1,2],[22,2,2],[14,3,2],[14,4,2],[6,5,2],[10,6,2], #FYNNEI
                [4,1,3],[3,2,3],[19,3,3],[22,4,3],[21,5,3]
            ]),
            torch.tensor([
                [2,0,0],
                [8,1,1],[19,2,1],[18,3,1],[14,4,1],[15,5,1],[14,6,1], # GTSNPN
                [18,1,2],[20,2,2],[8,3,2],[10,4,2],[8,5,2], # SVGIG
                [4,1,3],[3,2,3],[16,3,3],[22,4,3],[7,5,3]
            ])
        ]

        for (_, item), target in zip(mock_data_beta_df.iterrows(), expected):
            assert torch.equal(tokeniser.tokenise(item), target)