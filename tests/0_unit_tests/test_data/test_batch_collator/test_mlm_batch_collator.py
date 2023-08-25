import pytest
import random
import torch

from src.data.batch_collator import MlmBatchCollator
from src.data.tokeniser.beta_cdr_tokeniser import BetaCdrTokeniser
from src.data.tcr_dataset import TcrDataset


def test_collate_fn(mock_batch, expected_masked_tcrs, expected_mlm_targets):
    tokeniser = BetaCdrTokeniser()
    batch_generator = MlmBatchCollator(tokeniser)

    random.seed(4)
    masked_tcrs, mlm_targets = batch_generator.collate_fn(mock_batch)

    assert torch.equal(masked_tcrs, expected_masked_tcrs)
    assert torch.equal(mlm_targets, expected_mlm_targets)


@pytest.fixture
def mock_batch(mock_data_df):
    dataset = TcrDataset(mock_data_df)
    return dataset[:]


@pytest.fixture
def expected_masked_tcrs():
    return torch.tensor(
        [
            [
                [2, 0, 0, 0],
                [18, 1, 5, 1],
                [14, 2, 5, 1],
                [9, 3, 5, 1],
                [1, 4, 5, 1],
                [22, 5, 5, 1],
                [7, 1, 6, 2],
                [22, 2, 6, 2],
                [1, 3, 6, 2],
                [14, 4, 6, 2],
                [1, 5, 6, 2],
                [10, 6, 6, 2],
                [4, 1, 12, 3],
                [1, 2, 12, 3],
                [18, 3, 12, 3],
                [18, 4, 12, 3],
                [5, 5, 12, 3],
                [17, 6, 12, 3],
                [3, 7, 12, 3],
                [16, 8, 12, 3],
                [15, 9, 12, 3],
                [16, 10, 12, 3],
                [9, 11, 12, 3],
                [7, 12, 12, 3],
            ],
            [
                [2, 0, 0, 0],
                [18, 1, 5, 1],
                [14, 2, 5, 1],
                [1, 3, 5, 1],
                [12, 4, 5, 1],
                [1, 5, 5, 1],
                [7, 1, 6, 2],
                [22, 2, 6, 2],
                [14, 3, 6, 2],
                [14, 4, 6, 2],
                [6, 5, 6, 2],
                [10, 6, 6, 2],
                [4, 1, 10, 3],
                [3, 2, 10, 3],
                [18, 3, 10, 3],
                [18, 4, 10, 3],
                [1, 5, 10, 3],
                [14, 6, 10, 3],
                [5, 7, 10, 3],
                [1, 8, 10, 3],
                [3, 9, 10, 3],
                [7, 10, 10, 3],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [2, 0, 0, 0],
                [13, 1, 5, 1],
                [14, 2, 5, 1],
                [9, 3, 5, 1],
                [8, 4, 5, 1],
                [22, 5, 5, 1],
                [18, 1, 6, 2],
                [20, 2, 6, 2],
                [3, 3, 6, 2],
                [3, 4, 6, 2],
                [1, 5, 6, 2],
                [10, 6, 6, 2],
                [4, 1, 12, 3],
                [1, 2, 12, 3],
                [18, 3, 12, 3],
                [18, 4, 12, 3],
                [5, 5, 12, 3],
                [17, 6, 12, 3],
                [1, 7, 12, 3],
                [16, 8, 12, 3],
                [15, 9, 12, 3],
                [16, 10, 12, 3],
                [9, 11, 12, 3],
                [7, 12, 12, 3],
            ],
        ]
    )


@pytest.fixture
def expected_mlm_targets():
    return torch.tensor(
        [
            [0, 0, 0, 0, 12, 0, 0, 0, 14, 0, 6, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 9, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 17, 0, 0, 0, 0],
            [0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
        ]
    )
