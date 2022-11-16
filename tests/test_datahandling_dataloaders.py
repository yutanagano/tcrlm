import pytest
from src.datahandling import dataloaders
import torch


class TestTCRDataLoader:
    def test_padding_collation(self, cdr3t_dataset):
        dataloader = dataloaders.TCRDataLoader(
            dataset=cdr3t_dataset,
            batch_size=3
        )

        expected = torch.tensor(
            [
                [
                    [3,1],[2,1],[17,1],[15,1],[21,1],[6,1],
                    [3,2],[2,2],[18,2],[21,2],[20,2]
                ],
                [
                    [3,1],[2,1],[17,1],[15,1],[21,1],[6,1],
                    [0,0],[0,0],[0, 0],[0, 0],[0, 0]
                ],
                [
                    [3,2],[2,2],[18,2],[21,2],[20,2],[0,0],
                    [0,0],[0,0],[0, 0],[0, 0],[0, 0]
                ]
            ]
        )

        first_batch = next(iter(dataloader))

        assert first_batch.equal(expected)