import pandas as pd
import pytest
from src.datahandling.dataloaders import TCRDataLoader
from src.modules import AtchleyEmbedder
import torch


@pytest.fixture
def cdr3t_dataloader(cdr3t_dataset):
    dl = TCRDataLoader(dataset=cdr3t_dataset, batch_size=3)
    return dl


class TestAtchleyEmbedder:
    def test_embed(self, cdr3t_dataloader):
        embedder = AtchleyEmbedder()

        expected = torch.tensor([
            [
                -0.2231,  0.0703, -0.4933, -0.0585, -0.3359,
                -0.3116,  0.0444,  0.5940, -0.2042,  0.3033
            ],
            [
                -0.3466,  0.1092, -0.7664, -0.0908, -0.5219, 
                 0.0000,  0.0000,  0.0000,  0.0000,  0.0000
            ],
            [
                 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                -0.4071,  0.0580,  0.7763, -0.2668,  0.3964
            ]
        ])

        batch = next(iter(cdr3t_dataloader))

        print(batch)

        torch.testing.assert_close(
            embedder(batch),
            expected,
            atol=0,
            rtol=0.001
        )