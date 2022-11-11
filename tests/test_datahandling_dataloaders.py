import pytest
from src.datahandling import dataloaders
from src.datahandling.datasets import TCRDataset
from src.datahandling.tokenisers import CDR3Tokeniser
import torch


@pytest.fixture
def cdr3t_dataset(mock_data_path):
    dataset = TCRDataset(
        data=mock_data_path,
        tokeniser=CDR3Tokeniser()
    )

    return dataset


class TestTCRDataLoader:
    def test_padding_collation(self, cdr3t_dataset):
        dataloader = dataloaders.TCRDataLoader(
            dataset=cdr3t_dataset,
            batch_size=3
        )

        expected = torch.tensor(
            [
                [
                    [3,0],[2,0],[17,0],[15,0],[21,0],[6,0],
                    [3,1],[2,1],[18,1],[21,1],[20,1]
                ],
                [
                    [3,0],[2,0],[17,0],[15,0],[21,0],[6,0],
                    [0,0],[0,0],[0, 0],[0, 0],[0, 0]
                ],
                [
                    [3,1],[2,1],[18,1],[21,1],[20,1],[0,0],
                    [0,0],[0,0],[0, 0],[0, 0],[0, 0]
                ]
            ]
        )

        first_batch = next(iter(dataloader))

        assert first_batch.equal(expected)