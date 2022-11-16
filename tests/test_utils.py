import pytest
from src import utils
import torch


class TestMaskedAveragePool:
    @pytest.mark.parametrize(
        ('x','padding_mask','expected'),
        (
            (
                torch.tensor(
                    [
                        [[5,0,2],[4,6,2],[7,3,5]],
                        [[3,4,1],[9,7,2],[7,8,6]]
                    ],
                    dtype=torch.int
                ),
                torch.tensor(
                    [
                        [0,0,1],[0,1,1]
                    ],
                    dtype=torch.int
                ),
                torch.tensor(
                    [
                        [4.5,3,2],[3,4,1]
                    ],
                    dtype=torch.float32
                )
            ),
            (
                torch.tensor(
                    [
                        [[3,7,1],[5,6,1],[7,2,1]],
                        [[3,4,1],[9,8,2],[0,8,6]]
                    ],
                    dtype=torch.int
                ),
                torch.tensor(
                    [
                        [0,0,0],[0,0,1]
                    ],
                    dtype=torch.int
                ),
                torch.tensor(
                    [
                        [5,5,1],[6,6,1.5]
                    ],
                    dtype=torch.float32
                )
            )
        )
    )
    def test_masked_average_pool(
        self,
        x,
        padding_mask,
        expected
    ):
        result = utils.masked_average_pool(
            x=x,
            padding_mask=padding_mask
        )
        torch.testing.assert_close(result, expected)