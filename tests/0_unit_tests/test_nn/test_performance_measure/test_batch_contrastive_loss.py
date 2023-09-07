import torch

from src.nn.performance_measure import BatchContrastiveLoss


BATCH_SIZE = 4
REPRESENTATION_DIM = 5


def test_loss():
    dummy_representations = torch.rand((BATCH_SIZE, REPRESENTATION_DIM))
    dummy_mask = torch.tensor(
        [
            [0,0,1,0],
            [0,0,0,1],
            [1,0,0,0],
            [0,1,0,0]
        ],
        dtype=torch.bool
    )
    temp = 0.05
    loss_fn = BatchContrastiveLoss(temp)

    result = loss_fn.forward(dummy_representations, dummy_mask)
    expected = alternate_loss_computation(dummy_representations, dummy_mask, temp)

    torch.testing.assert_close(result, expected)


def alternate_loss_computation(representations, mask, temp):
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