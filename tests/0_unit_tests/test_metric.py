import pytest
from src.nn.performance_measure import *
import torch


class TestAdjustedCELoss:
    def test_init(self):
        loss_fn = AdjustedCELoss(label_smoothing=0.5)

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
    def test_forward(self, y, expected):
        loss_fn = AdjustedCELoss()
        x = torch.tensor([[0.5, 0.2, 0.3], [0.3, 0.3, 0.4]])

        result = loss_fn(x, y)

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("token", (1, 6, -100))
    def test_error_padding_tokens(self, token):
        loss_fn = AdjustedCELoss()
        x = torch.tensor([[0.5, 0.2, 0.3]])

        with pytest.raises(IndexError):
            loss_fn(x, torch.tensor([token]))


class TestBatchContrastiveLoss:
    BATCH_SIZE = 4
    REPRESENTATION_DIM = 5

    def test_loss(self):
        dummy_representations = torch.rand((self.BATCH_SIZE, self.REPRESENTATION_DIM))
        dummy_mask = torch.tensor(
            [
                [0,0,1,0],
                [0,0,0,1],
                [1,0,0,0],
                [0,1,0,0]
            ],
            dtype=torch.long
        )
        temp = 0.05
        loss_fn = BatchContrastiveLoss(temp)

        result = loss_fn.forward(dummy_representations, dummy_mask)
        expected = self.alternate_loss_computation(dummy_representations, dummy_mask, temp)

        torch.testing.assert_close(result, expected)

    def alternate_loss_computation(self, representations, mask, temp):
        dot_products = torch.matmul(representations, representations.T) / temp
        exp_dot_products = torch.exp(dot_products)

        identity_mask = torch.eye(len(representations))
        non_identity_mask = torch.logical_not(identity_mask)
        denominator = torch.sum(exp_dot_products * non_identity_mask, dim=1)

        fraction = exp_dot_products / denominator
        logged_fraction = torch.log(fraction)

        num_positives_per_sample = torch.sum(mask, dim=1)
        contributions_from_positive_terms = torch.sum(logged_fraction * mask, dim=1)

        loss_per_sample = -contributions_from_positive_terms / num_positives_per_sample

        return loss_per_sample.mean()


class TestMLMAccuracy:
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
    def test_mlm_accuracy(self, logits, y, expected):
        calculated = mlm_acc(logits, y)
        torch.testing.assert_close(calculated, expected)


class TestMLMTopkAccuracy:
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
    def test_mlm_topk_accuracy(self, logits, y, k, expected):
        calculated = mlm_topk_acc(logits, y, k)
        torch.testing.assert_close(calculated, expected)