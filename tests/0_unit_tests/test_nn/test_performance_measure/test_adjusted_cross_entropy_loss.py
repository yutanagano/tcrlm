import pytest
import torch

from src.nn.performance_measure import AdjustedCrossEntropyLoss


def test_init():
    loss_fn = AdjustedCrossEntropyLoss(label_smoothing=0.5)

    assert loss_fn.label_smoothing == 0.5
    assert loss_fn.ignore_index == -3


@pytest.mark.parametrize(
    ("y", "expected"),
    (
        (torch.tensor([4, 4]), torch.tensor(1.1864500045776367)),
        (torch.tensor([0, 3]), torch.tensor(1.1330687999725342)),
        (torch.tensor([5, 0]), torch.tensor(1.1398310661315918)),
    ),
)
def test_forward(y, expected):
    loss_fn = AdjustedCrossEntropyLoss(label_smoothing=0)
    x = torch.tensor([[0.5, 0.2, 0.3], [0.3, 0.3, 0.4]])

    result = loss_fn(x, y)

    print(result)
    print(expected)

    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("token", (1, 6, -100))
def test_error_padding_tokens(token):
    loss_fn = AdjustedCrossEntropyLoss()
    x = torch.tensor([[0.5, 0.2, 0.3]])

    with pytest.raises(IndexError):
        loss_fn(x, torch.tensor([token]))
