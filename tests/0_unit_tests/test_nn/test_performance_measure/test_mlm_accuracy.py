import pytest
import torch

from src.nn.performance_measure import mlm_acc, mlm_topk_acc


@pytest.mark.parametrize(
    ("logits", "y", "expected"),
    (
        (
            torch.tensor(
                [
                    [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],
                    [[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]],
                ],
                dtype=torch.float,
            ),
            torch.tensor([[3, 4, 5], [7, 6, 5]], dtype=torch.long),
            4 / 6,
        ),
        (
            torch.tensor(
                [
                    [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],
                    [[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]],
                ],
                dtype=torch.float,
            ),
            torch.tensor([[3, 4, 5], [7, 0, 0]], dtype=torch.long),
            3 / 4,
        ),
    ),
)
def test_mlm_accuracy(logits, y, expected):
    calculated = mlm_acc(logits, y)
    torch.testing.assert_close(calculated, expected)


@pytest.mark.parametrize(
    ("logits", "y", "k", "expected"),
    (
        (
            torch.tensor(
                [
                    [
                        [0.2, 0.5, 0.3, 0, 0],
                        [0.3, 0.5, 0.2, 0, 0],
                        [0.2, 0.5, 0.3, 0, 0],
                    ],
                    [
                        [0.2, 0, 0.3, 0, 0.5],
                        [0, 0.3, 0, 0.2, 0.5],
                        [0, 0, 0.5, 0.3, 0.2],
                    ],
                ],
                dtype=torch.float,
            ),
            torch.tensor([[3, 4, 5], [7, 6, 5]], dtype=torch.long),
            2,
            4 / 6,
        ),
        (
            torch.tensor(
                [
                    [
                        [0.2, 0.5, 0.3, 0, 0],
                        [0.3, 0.5, 0.2, 0, 0],
                        [0.2, 0.5, 0.3, 0, 0],
                    ],
                    [
                        [0.2, 0, 0.3, 0, 0.5],
                        [0, 0.3, 0, 0.2, 0.5],
                        [0, 0, 0.5, 0.3, 0.2],
                    ],
                ],
                dtype=torch.float,
            ),
            torch.tensor([[3, 4, 5], [7, 0, 0]], dtype=torch.long),
            2,
            3 / 4,
        ),
    ),
)
def test_mlm_topk_accuracy(logits, y, k, expected):
    calculated = mlm_topk_acc(logits, y, k)
    torch.testing.assert_close(calculated, expected)
