import pytest
import random
import torch

from src.nn.data.batch_collator import ClBatchCollator
from src.nn.data.tokeniser.beta_cdr_tokeniser import BetaCdrTokeniser
from src.nn.data.tcr_dataset import TcrDataset


def test_collate_fn(
    mock_batch,
    expected_double_view_batch,
    expected_double_view_positives_mask,
    expected_masked_tcrs,
    expected_mlm_targets,
):
    tokeniser = BetaCdrTokeniser()
    batch_generator = ClBatchCollator(tokeniser)

    random.seed(4)
    (
        double_view_batch,
        double_view_positives_mask,
        masked_tcrs,
        mlm_targets,
    ) = batch_generator.collate_fn(mock_batch)

    assert torch.equal(double_view_batch, expected_double_view_batch)
    assert torch.equal(double_view_positives_mask, expected_double_view_positives_mask)
    assert torch.equal(masked_tcrs, expected_masked_tcrs)
    assert torch.equal(mlm_targets, expected_mlm_targets)


@pytest.fixture
def mock_batch(mock_data_df):
    dataset = TcrDataset(mock_data_df)
    return dataset[:]


@pytest.fixture
def expected_double_view_batch():
    return torch.tensor(
        [
            [
                [2, 0, 0, 0],
                [18, 1, 5, 1],
                [14, 2, 5, 1],
                [9, 3, 5, 1],
                [0, 4, 5, 1],
                [22, 5, 5, 1],
                [7, 1, 6, 2],
                [22, 2, 6, 2],
                [0, 3, 6, 2],
                [14, 4, 6, 2],
                [0, 5, 6, 2],
                [10, 6, 6, 2],
                [4, 1, 12, 3],
                [0, 2, 12, 3],
                [18, 3, 12, 3],
                [18, 4, 12, 3],
                [0, 5, 12, 3],
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
                [0, 1, 5, 1],
                [14, 2, 5, 1],
                [0, 3, 5, 1],
                [12, 4, 5, 1],
                [0, 5, 5, 1],
                [7, 1, 6, 2],
                [22, 2, 6, 2],
                [14, 3, 6, 2],
                [14, 4, 6, 2],
                [6, 5, 6, 2],
                [10, 6, 6, 2],
                [4, 1, 10, 3],
                [0, 2, 10, 3],
                [18, 3, 10, 3],
                [18, 4, 10, 3],
                [3, 5, 10, 3],
                [14, 6, 10, 3],
                [5, 7, 10, 3],
                [17, 8, 10, 3],
                [0, 9, 10, 3],
                [7, 10, 10, 3],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [2, 0, 0, 0],
                [13, 1, 5, 1],
                [0, 2, 5, 1],
                [9, 3, 5, 1],
                [8, 4, 5, 1],
                [22, 5, 5, 1],
                [18, 1, 6, 2],
                [20, 2, 6, 2],
                [0, 3, 6, 2],
                [3, 4, 6, 2],
                [0, 5, 6, 2],
                [10, 6, 6, 2],
                [4, 1, 12, 3],
                [3, 2, 12, 3],
                [18, 3, 12, 3],
                [18, 4, 12, 3],
                [5, 5, 12, 3],
                [0, 6, 12, 3],
                [0, 7, 12, 3],
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
                [9, 3, 5, 1],
                [0, 4, 5, 1],
                [22, 5, 5, 1],
                [0, 1, 6, 2],
                [22, 2, 6, 2],
                [14, 3, 6, 2],
                [0, 4, 6, 2],
                [6, 5, 6, 2],
                [10, 6, 6, 2],
                [0, 1, 12, 3],
                [3, 2, 12, 3],
                [18, 3, 12, 3],
                [18, 4, 12, 3],
                [5, 5, 12, 3],
                [17, 6, 12, 3],
                [0, 7, 12, 3],
                [16, 8, 12, 3],
                [15, 9, 12, 3],
                [16, 10, 12, 3],
                [9, 11, 12, 3],
                [7, 12, 12, 3],
            ],
            [
                [2, 0, 0, 0],
                [0, 1, 5, 1],
                [14, 2, 5, 1],
                [9, 3, 5, 1],
                [12, 4, 5, 1],
                [22, 5, 5, 1],
                [7, 1, 6, 2],
                [0, 2, 6, 2],
                [14, 3, 6, 2],
                [0, 4, 6, 2],
                [6, 5, 6, 2],
                [10, 6, 6, 2],
                [4, 1, 10, 3],
                [3, 2, 10, 3],
                [18, 3, 10, 3],
                [18, 4, 10, 3],
                [3, 5, 10, 3],
                [14, 6, 10, 3],
                [0, 7, 10, 3],
                [17, 8, 10, 3],
                [3, 9, 10, 3],
                [0, 10, 10, 3],
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
                [0, 1, 6, 2],
                [0, 2, 6, 2],
                [3, 3, 6, 2],
                [3, 4, 6, 2],
                [0, 5, 6, 2],
                [10, 6, 6, 2],
                [0, 1, 12, 3],
                [3, 2, 12, 3],
                [18, 3, 12, 3],
                [18, 4, 12, 3],
                [5, 5, 12, 3],
                [17, 6, 12, 3],
                [3, 7, 12, 3],
                [16, 8, 12, 3],
                [15, 9, 12, 3],
                [0, 10, 12, 3],
                [9, 11, 12, 3],
                [7, 12, 12, 3],
            ],
        ]
    )


@pytest.fixture
def expected_double_view_positives_mask():
    return torch.tensor(
        [
            [0,1,0,1,1,0],
            [1,0,0,1,1,0],
            [0,0,0,0,0,1],
            [1,1,0,0,1,0],
            [1,1,0,1,0,0],
            [0,0,1,0,0,0],
        ]
    )


@pytest.fixture
def expected_masked_tcrs():
    return torch.tensor(
        [
            [
                [2, 0, 0, 0],
                [18, 1, 5, 1],
                [14, 2, 5, 1],
                [9, 3, 5, 1],
                [12, 4, 5, 1],
                [22, 5, 5, 1],
                [7, 1, 6, 2],
                [22, 2, 6, 2],
                [14, 3, 6, 2],
                [14, 4, 6, 2],
                [6, 5, 6, 2],
                [10, 6, 6, 2],
                [4, 1, 12, 3],
                [3, 2, 12, 3],
                [18, 3, 12, 3],
                [18, 4, 12, 3],
                [5, 5, 12, 3],
                [17, 6, 12, 3],
                [3, 7, 12, 3],
                [16, 8, 12, 3],
                [1, 9, 12, 3],
                [16, 10, 12, 3],
                [1, 11, 12, 3],
                [7, 12, 12, 3],
            ],
            [
                [2, 0, 0, 0],
                [18, 1, 5, 1],
                [14, 2, 5, 1],
                [9, 3, 5, 1],
                [12, 4, 5, 1],
                [22, 5, 5, 1],
                [1, 1, 6, 2],
                [22, 2, 6, 2],
                [1, 3, 6, 2],
                [14, 4, 6, 2],
                [6, 5, 6, 2],
                [10, 6, 6, 2],
                [4, 1, 10, 3],
                [12, 2, 10, 3],
                [18, 3, 10, 3],
                [18, 4, 10, 3],
                [3, 5, 10, 3],
                [1, 6, 10, 3],
                [5, 7, 10, 3],
                [17, 8, 10, 3],
                [3, 9, 10, 3],
                [7, 10, 10, 3],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [2, 0, 0, 0],
                [13, 1, 5, 1],
                [14, 2, 5, 1],
                [1, 3, 5, 1],
                [8, 4, 5, 1],
                [22, 5, 5, 1],
                [18, 1, 6, 2],
                [20, 2, 6, 2],
                [1, 3, 6, 2],
                [1, 4, 6, 2],
                [8, 5, 6, 2],
                [10, 6, 6, 2],
                [4, 1, 12, 3],
                [3, 2, 12, 3],
                [18, 3, 12, 3],
                [18, 4, 12, 3],
                [1, 5, 12, 3],
                [17, 6, 12, 3],
                [3, 7, 12, 3],
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
            [0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 9, 0],
            [0, 0, 0, 0, 0, 0, 7, 0, 14, 0, 0, 0, 0, 3, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 9, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
