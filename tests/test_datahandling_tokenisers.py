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
                [3,1],[2,1],[17,1],[15,1],[21,1],[6,1],
                [3,2],[2,2],[18,2],[21,2],[20,2]
            ]),
            torch.tensor([
                [3,1],[2,1],[17,1],[15,1],[21,1],[6,1]
            ]),
            torch.tensor([
                [3,2],[2,2],[18,2],[21,2],[20,2]
            ]),
        ]

        for (_, item), target in zip(mock_data_df.iterrows(), expected):
            assert torch.equal(tokeniser.tokenise(item), target)