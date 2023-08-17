import pytest
import random
import torch

from src.model_trainer.batch_collator import AclBatchCollator
from src.data.tokeniser.beta_cdr_tokeniser import BetaCdrTokeniser


def test_collate_fn(mock_tokenised_tcrs, expected_anchor_tcrs, expected_positive_pair_tcrs, expected_masked_tcrs, expected_mlm_targets):
    tokeniser = BetaCdrTokeniser()
    batch_generator = AclBatchCollator(tokeniser)

    random.seed(4)
    anchor_tcrs, positive_pair_tcrs, masked_tcrs, mlm_targets = batch_generator.collate_fn(mock_tokenised_tcrs)

    assert torch.equal(anchor_tcrs, expected_anchor_tcrs)
    assert torch.equal(positive_pair_tcrs, expected_positive_pair_tcrs)
    assert torch.equal(masked_tcrs, expected_masked_tcrs)
    assert torch.equal(mlm_targets, expected_mlm_targets)


@pytest.fixture
def mock_tokenised_tcrs():
    return [
        torch.tensor(
            [
                [2, 0, 0, 0],
                [7, 1, 5, 1],
                [7, 2, 5, 1],
                [7, 3, 5, 1],
                [7, 4, 5, 1],
                [7, 5, 5, 1],
                [7, 1, 5, 2],
                [7, 2, 5, 2],
                [7, 3, 5, 2],
                [7, 4, 5, 2],
                [7, 5, 5, 2],
                [7, 1, 5, 3],
                [7, 2, 5, 3],
                [7, 3, 5, 3],
                [7, 4, 5, 3],
                [7, 5, 5, 3],
            ]
        ),
        torch.tensor(
            [
                [2, 0, 0, 0],
                [7, 1, 4, 1],
                [7, 2, 4, 1],
                [7, 3, 4, 1],
                [7, 4, 4, 1],
                [7, 1, 4, 2],
                [7, 2, 4, 2],
                [7, 3, 4, 2],
                [7, 4, 4, 2],
                [7, 1, 4, 3],
                [7, 2, 4, 3],
                [7, 3, 4, 3],
                [7, 4, 4, 3],
            ]
        ),
    ]


@pytest.fixture
def expected_anchor_tcrs():
    return torch.tensor(
        [
            [
                [2, 0, 0, 0],
                [7, 1, 5, 1],
                [0, 2, 5, 1],
                [7, 3, 5, 1],
                [0, 4, 5, 1],
                [0, 5, 5, 1],
                [7, 1, 5, 2],
                [7, 2, 5, 2],
                [7, 3, 5, 2],
                [7, 4, 5, 2],
                [7, 5, 5, 2],
                [7, 1, 5, 3],
                [7, 2, 5, 3],
                [7, 3, 5, 3],
                [7, 4, 5, 3],
                [7, 5, 5, 3],
            ],
            [
                [2, 0, 0, 0],
                [7, 1, 4, 1],
                [7, 2, 4, 1],
                [7, 3, 4, 1],
                [7, 4, 4, 1],
                [7, 1, 4, 2],
                [7, 2, 4, 2],
                [0, 3, 4, 2],
                [0, 4, 4, 2],
                [7, 1, 4, 3],
                [7, 2, 4, 3],
                [7, 3, 4, 3],
                [0, 4, 4, 3],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ]
    )


@pytest.fixture
def expected_positive_pair_tcrs():
    return torch.tensor(
        [
            [
                [2, 0, 0, 0],
                [7, 1, 5, 1],
                [0, 2, 5, 1],
                [0, 3, 5, 1],
                [7, 4, 5, 1],
                [7, 5, 5, 1],
                [7, 1, 5, 2],
                [7, 2, 5, 2],
                [7, 3, 5, 2],
                [7, 4, 5, 2],
                [7, 5, 5, 2],
                [7, 1, 5, 3],
                [7, 2, 5, 3],
                [7, 3, 5, 3],
                [0, 4, 5, 3],
                [7, 5, 5, 3],
            ],
            [
                [2, 0, 0, 0],
                [0, 1, 4, 1],
                [7, 2, 4, 1],
                [7, 3, 4, 1],
                [7, 4, 4, 1],
                [7, 1, 4, 2],
                [7, 2, 4, 2],
                [0, 3, 4, 2],
                [7, 4, 4, 2],
                [0, 1, 4, 3],
                [7, 2, 4, 3],
                [7, 3, 4, 3],
                [7, 4, 4, 3],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ]
    )


@pytest.fixture
def expected_masked_tcrs():
    return torch.tensor(
        [
            [
                [2, 0, 0, 0],
                [7, 1, 5, 1],
                [7, 2, 5, 1],
                [7, 3, 5, 1],
                [7, 4, 5, 1],
                [1, 5, 5, 1],
                [7, 1, 5, 2],
                [7, 2, 5, 2],
                [7, 3, 5, 2],
                [7, 4, 5, 2],
                [7, 5, 5, 2],
                [7, 1, 5, 3],
                [7, 2, 5, 3],
                [1, 3, 5, 3],
                [7, 4, 5, 3],
                [1, 5, 5, 3],
            ],
            [
                [2, 0, 0, 0],
                [1, 1, 4, 1],
                [7, 2, 4, 1],
                [7, 3, 4, 1],
                [1, 4, 4, 1],
                [7, 1, 4, 2],
                [7, 2, 4, 2],
                [7, 3, 4, 2],
                [7, 4, 4, 2],
                [7, 1, 4, 3],
                [7, 2, 4, 3],
                [7, 3, 4, 3],
                [7, 4, 4, 3],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ]
    )


@pytest.fixture
def expected_mlm_targets():
    return torch.tensor(
        [
            [0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 7],
            [0, 7, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
