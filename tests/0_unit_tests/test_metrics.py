import pytest
from src import metrics
import torch


class TestAlignment:
    def test_alignment(self):
        x = torch.tensor(
            [
                [ 0, 1],
                [ 1, 0],
                [ 0,-1],
                [-1, 0]
            ],
            dtype=torch.float32
        )
        labels = torch.tensor([0,0,1,1])

        torch.testing.assert_close(
            metrics.alignment(x,labels,alpha=2),
            torch.tensor(2, dtype=torch.float32)
        )


class TestUniformity:
    def test_uniformity(self):
        x = torch.tensor(
            [
                [ 0, 1],
                [ 1, 0],
                [ 0,-1],
                [-1, 0]
            ],
            dtype=torch.float32
        )

        torch.testing.assert_close(
            metrics.uniformity(x,t=2),
            torch.log(torch.tensor(1/3)) + \
                torch.log(
                    torch.tensor(2.0) + torch.exp(torch.tensor(-4.0))
                ) - \
                torch.tensor(4.0)
        )


class TestAdjustedCELoss:
    def test_init(self):
        criterion = metrics.AdjustedCELoss(label_smoothing=0.5)

        assert criterion.label_smoothing == 0.5
        assert criterion.ignore_index == -3


    @pytest.mark.parametrize(
        ('y', 'expected'),
        (
            (torch.tensor([4,4]), torch.tensor(1.1864500045776367)),
            (torch.tensor([0,3]), torch.tensor(1.1330687999725342)),
            (torch.tensor([5,0]), torch.tensor(1.1398310661315918))
        )
    )
    def test_forward(self, y, expected):
        criterion = metrics.AdjustedCELoss()
        x = torch.tensor([[0.5,0.2,0.3],[0.3,0.3,0.4]])

        result = criterion(x, y)

        torch.testing.assert_close(result, expected)


    @pytest.mark.parametrize(
        'token', (1,6,-100)
    )
    def test_error_padding_tokens(self, token):
        criterion = metrics.AdjustedCELoss()
        x = torch.tensor([[0.5,0.2,0.3]])

        with pytest.raises(IndexError):
            criterion(x, torch.tensor([token]))


class TestMLMAccuracy:
    @pytest.mark.parametrize(
        ('logits', 'y', 'expected'),
        (
            (
                torch.tensor(
                    [[[0,1,0,0,0],[0,1,0,0,0],[0,0,1,0,0]],
                     [[0,0,0,0,1],[0,0,0,0,1],[0,0,1,0,0]]],
                    dtype=torch.float
                ),
                torch.tensor([[3,4,5],[7,6,5]], dtype=torch.long),
                4/6
            ),
            (
                torch.tensor(
                    [[[0,1,0,0,0],[0,1,0,0,0],[0,0,1,0,0]],
                     [[0,0,0,0,1],[0,0,0,0,1],[0,0,1,0,0]]],
                    dtype=torch.float
                ),
                torch.tensor([[3,4,5],[7,0,0]], dtype=torch.long),
                3/4
            )
        )
    )
    def test_mlm_accuracy(self, logits, y, expected):
        calculated = metrics.mlm_acc(logits, y)
        torch.testing.assert_close(calculated, expected)


class TestMLMTopkAccuracy:
    @pytest.mark.parametrize(
        ('logits', 'y', 'k', 'expected'),
        (
            (
                torch.tensor(
                    [[[0.2,0.5,0.3,0,0],[0.3,0.5,0.2,0,0],[0.2,0.5,0.3,0,0]],
                     [[0.2,0,0.3,0,0.5],[0,0.3,0,0.2,0.5],[0,0,0.5,0.3,0.2]]],
                    dtype=torch.float
                ),
                torch.tensor([[3,4,5],[7,6,5]], dtype=torch.long),
                2,
                4/6
            ),
            (
                torch.tensor(
                    [[[0.2,0.5,0.3,0,0],[0.3,0.5,0.2,0,0],[0.2,0.5,0.3,0,0]],
                     [[0.2,0,0.3,0,0.5],[0,0.3,0,0.2,0.5],[0,0,0.5,0.3,0.2]]],
                    dtype=torch.float
                ),
                torch.tensor([[3,4,5],[7,0,0]], dtype=torch.long),
                2,
                3/4
            )
        )
    )
    def test_mlm_topk_accuracy(self, logits, y, k, expected):
        calculated = metrics.mlm_topk_acc(logits, y, k)
        torch.testing.assert_close(calculated, expected)