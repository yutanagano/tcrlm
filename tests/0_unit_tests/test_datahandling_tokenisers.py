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


class TestCDR3Tokeniser:
    def test_tokenise(self, mock_data_df):
        tokeniser = tokenisers.CDR3ABTokeniser()

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
        tokeniser = tokenisers.CDR3ABTokeniser()

        for (_, item), target in zip(mock_data_df.iterrows(), expected):
            assert torch.equal(tokeniser.tokenise(item, chain), target)


    def test_vocab_size(self):
        tokeniser = tokenisers.CDR3ABTokeniser()

        assert tokeniser.vocab_size == 20


class TestCDR3BetaTokeniser:
    def test_tokenise(self, mock_data_beta_df):
        tokeniser = tokenisers.CDR3BTokeniser()

        expected = [
            torch.tensor([
                [2,0],
                [4,1],[3,2],[19,3],[22,4],[21,5]
            ]),
            torch.tensor([
                [2,0],
                [4,1],[3,2],[19,3],[22,4],[21,5]
            ]),
        ]

        for (_, item), target in zip(mock_data_beta_df.iterrows(), expected):
            assert torch.equal(tokeniser.tokenise(item), target)


    def test_vocab_size(self):
        tokeniser = tokenisers.CDR3ABTokeniser()

        assert tokeniser.vocab_size == 20