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
                    [3,0],[2,0],[17,0],[15,0],[21,0],[6,0],
                    [3,1],[2,1],[18,1],[21,1],[20,1]
            ]),
            torch.tensor([
                    [3,0],[2,0],[17,0],[15,0],[21,0],[6,0]
            ]),
            torch.tensor([
                    [3,1],[2,1],[18,1],[21,1],[20,1]
            ]),
        ]

        for (_, item), target in zip(mock_data_df.iterrows(), expected):
            print(item)
            assert torch.equal(tokeniser.tokenise(item), target)