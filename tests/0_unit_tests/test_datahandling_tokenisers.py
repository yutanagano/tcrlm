import pandas as pd
import pytest
from src.datahandling import tokenisers
import torch


@pytest.fixture
def mock_data_df():
    df = pd.read_csv('tests/resources/mock_data.csv', dtype='string')
    return df


class TestCDR3Tokeniser:
    def test_tokenise(self, mock_data_df):
        tokeniser = tokenisers.CDR3Tokeniser()

        expected = [
            torch.tensor([
                [2,0,0],
                [4,1,1],[3,1,2],[18,1,3],[16,1,4],[22,1,5],[7,1,6],
                [4,2,1],[3,2,2],[19,2,3],[22,2,4],[21,2,5]
            ]),
            torch.tensor([
                [2,0,0],
                [4,1,1],[3,1,2],[18,1,3],[16,1,4],[22,1,5],[7,1,6]
            ]),
            torch.tensor([
                [2,0,0],
                [4,2,1],[3,2,2],[19,2,3],[22,2,4],[21,2,5]
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
                        [4,1,1],[3,1,2],[18,1,3],[16,1,4],[22,1,5],[7,1,6]
                    ]),
                    torch.tensor([
                        [2,0,0],
                        [4,1,1],[3,1,2],[18,1,3],[16,1,4],[22,1,5],[7,1,6]
                    ]),
                    torch.tensor([
                        [2,0,0],
                        [4,2,1],[3,2,2],[19,2,3],[22,2,4],[21,2,5]
                    ]),
                ]
            ),
            (
                'beta',
                [
                    torch.tensor([
                        [2,0,0],
                        [4,2,1],[3,2,2],[19,2,3],[22,2,4],[21,2,5]
                    ]),
                    torch.tensor([
                        [2,0,0],
                        [4,1,1],[3,1,2],[18,1,3],[16,1,4],[22,1,5],[7,1,6]
                    ]),
                    torch.tensor([
                        [2,0,0],
                        [4,2,1],[3,2,2],[19,2,3],[22,2,4],[21,2,5]
                    ]),
                ]
            ),
        )
    )
    def test_chain(self, mock_data_df, chain, expected):
        tokeniser = tokenisers.CDR3Tokeniser()

        for (_, item), target in zip(mock_data_df.iterrows(), expected):
            assert torch.equal(tokeniser.tokenise(item, chain), target)


    def test_vocab_size(self):
        tokeniser = tokenisers.CDR3Tokeniser()

        assert tokeniser.vocab_size == 20