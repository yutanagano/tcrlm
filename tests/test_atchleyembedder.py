import pandas as pd
import pytest
from src.datahandling.datasets import TCRDataset
from src.datahandling.tokenisers import CDR3Tokeniser
from src.modules import AtchleyEmbedder
import torch


@pytest.fixture
def mock_data():
    df = pd.read_csv('tests/resources/mock_data.csv', dtype='string')
    ds = TCRDataset(df, tokeniser=CDR3Tokeniser())
    return ds


class TestAtchleyEmbedder:
    def test_embed(self, mock_data):
        embedder = AtchleyEmbedder()

        expected = [
            torch.tensor([[
                -0.2231,  0.0703, -0.4933, -0.0585, -0.3359,
                -0.3116,  0.0444,  0.5940, -0.2042,  0.3033
            ]]),
            torch.tensor([[
                -0.3466,  0.1092, -0.7664, -0.0908, -0.5219, 
                 0.0000,  0.0000,  0.0000,  0.0000,  0.0000
            ]]),
            torch.tensor([[
                 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                -0.4071,  0.0580,  0.7763, -0.2668,  0.3964
            ]])
        ]

        for item, target in zip(mock_data, expected):
            item = item.unsqueeze(0)

            torch.testing.assert_close(
                embedder(item),
                target,
                atol=0,
                rtol=0.01
            )