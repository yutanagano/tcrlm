import pytest
import random
import torch

from src.nn.data.batch_collator import ClBatchCollator
from src.nn.data.tokeniser import CdrTokeniser
from src.nn.data.tcr_dataset import TcrDataset


def test_collate_fn(
    mock_batch,
    expected_double_view_batch,
    expected_double_view_positives_mask,
    expected_masked_tcrs,
    expected_mlm_targets,
):
    tokeniser = CdrTokeniser()
    batch_generator = ClBatchCollator(tokeniser)

    random.seed(420)
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
                [ 2,  0,  0,  0],
                [ 0,  1,  6,  1],
                [18,  2,  6,  1],
                [ 8,  3,  6,  1],
                [ 0,  4,  6,  1],
                [22,  5,  6,  1],
                [ 8,  6,  6,  1],
                [14,  1,  6,  2],
                [ 3,  2,  6,  2],
                [12,  3,  6,  2],
                [ 5,  4,  6,  2],
                [ 0,  5,  6,  2],
                [12,  6,  6,  2],
                [ 4,  1, 11,  3],
                [ 3,  2, 11,  3],
                [20,  3, 11,  3],
                [11,  4, 11,  3],
                [ 0,  5, 11,  3],
                [18,  6, 11,  3],
                [ 8,  7, 11,  3],
                [18,  8, 11,  3],
                [17,  9, 11,  3],
                [ 0, 10, 11,  3],
                [19, 11, 11,  3],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0]
            ],
            [
                [ 2,  0,  0,  0],
                [ 0,  1,  5,  4],
                [14,  2,  5,  4],
                [ 9,  3,  5,  4],
                [12,  4,  5,  4],
                [ 0,  5,  5,  4],
                [ 7,  1,  6,  5],
                [22,  2,  6,  5],
                [14,  3,  6,  5],
                [14,  4,  6,  5],
                [ 6,  5,  6,  5],
                [ 0,  6,  6,  5],
                [ 4,  1, 10,  6],
                [ 3,  2, 10,  6],
                [18,  3, 10,  6],
                [18,  4, 10,  6],
                [ 0,  5, 10,  6],
                [14,  6, 10,  6],
                [ 0,  7, 10,  6],
                [17,  8, 10,  6],
                [ 3,  9, 10,  6],
                [ 7, 10, 10,  6],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0]
            ],
            [
                [ 2,  0,  0,  0],
                [ 5,  1,  6,  1],
                [18,  2,  6,  1],
                [18,  3,  6,  1],
                [18,  4,  6,  1],
                [19,  5,  6,  1],
                [22,  6,  6,  1],
                [ 0,  1,  7,  2],
                [ 0,  2,  7,  2],
                [18,  3,  7,  2],
                [ 0,  4,  7,  2],
                [13,  5,  7,  2],
                [ 0,  6,  7,  2],
                [13,  7,  7,  2],
                [ 4,  1, 11,  3],
                [ 3,  2, 11,  3],
                [20,  3, 11,  3],
                [ 0,  4, 11,  3],
                [ 0,  5, 11,  3],
                [18,  6, 11,  3],
                [ 8,  7, 11,  3],
                [18,  8, 11,  3],
                [ 0,  9, 11,  3],
                [12, 10, 11,  3],
                [19, 11, 11,  3],
                [13,  1,  5,  4],
                [14,  2,  5,  4],
                [ 9,  3,  5,  4],
                [ 0,  4,  5,  4],
                [22,  5,  5,  4],
                [18,  1,  6,  5],
                [20,  2,  6,  5],
                [ 0,  3,  6,  5],
                [ 3,  4,  6,  5],
                [ 8,  5,  6,  5],
                [10,  6,  6,  5],
                [ 4,  1, 12,  6],
                [ 3,  2, 12,  6],
                [18,  3, 12,  6],
                [18,  4, 12,  6],
                [ 5,  5, 12,  6],
                [17,  6, 12,  6],
                [ 3,  7, 12,  6],
                [16,  8, 12,  6],
                [15,  9, 12,  6],
                [16, 10, 12,  6],
                [ 0, 11, 12,  6],
                [ 7, 12, 12,  6]
            ],
            [
                [ 2,  0,  0,  0],
                [ 0,  1,  5,  4],
                [ 0,  2,  5,  4],
                [ 0,  3,  5,  4],
                [12,  4,  5,  4],
                [22,  5,  5,  4],
                [ 7,  1,  6,  5],
                [22,  2,  6,  5],
                [14,  3,  6,  5],
                [14,  4,  6,  5],
                [ 6,  5,  6,  5],
                [10,  6,  6,  5],
                [ 4,  1, 12,  6],
                [ 3,  2, 12,  6],
                [18,  3, 12,  6],
                [18,  4, 12,  6],
                [ 5,  5, 12,  6],
                [ 0,  6, 12,  6],
                [ 3,  7, 12,  6],
                [16,  8, 12,  6],
                [15,  9, 12,  6],
                [16, 10, 12,  6],
                [ 0, 11, 12,  6],
                [ 7, 12, 12,  6],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0]
            ],
            [
                [ 2,  0,  0,  0],
                [18,  1,  5,  4],
                [14,  2,  5,  4],
                [ 9,  3,  5,  4],
                [12,  4,  5,  4],
                [ 0,  5,  5,  4],
                [ 7,  1,  6,  5],
                [ 0,  2,  6,  5],
                [14,  3,  6,  5],
                [14,  4,  6,  5],
                [ 0,  5,  6,  5],
                [10,  6,  6,  5],
                [ 4,  1, 10,  6],
                [ 3,  2, 10,  6],
                [18,  3, 10,  6],
                [18,  4, 10,  6],
                [ 3,  5, 10,  6],
                [ 0,  6, 10,  6],
                [ 0,  7, 10,  6],
                [17,  8, 10,  6],
                [ 3,  9, 10,  6],
                [ 7, 10, 10,  6],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0]
            ],
            [
                [ 2,  0,  0,  0],
                [ 0,  1,  6,  1],
                [ 0,  2,  6,  1],
                [18,  3,  6,  1],
                [18,  4,  6,  1],
                [19,  5,  6,  1],
                [22,  6,  6,  1],
                [10,  1,  7,  2],
                [ 0,  2,  7,  2],
                [ 0,  3,  7,  2],
                [14,  4,  7,  2],
                [13,  5,  7,  2],
                [ 5,  6,  7,  2],
                [ 0,  7,  7,  2],
                [ 4,  1, 11,  3],
                [ 3,  2, 11,  3],
                [20,  3, 11,  3],
                [11,  4, 11,  3],
                [ 0,  5, 11,  3],
                [18,  6, 11,  3],
                [ 8,  7, 11,  3],
                [18,  8, 11,  3],
                [17,  9, 11,  3],
                [12, 10, 11,  3],
                [19, 11, 11,  3],
                [13,  1,  5,  4],
                [14,  2,  5,  4],
                [ 0,  3,  5,  4],
                [ 8,  4,  5,  4],
                [22,  5,  5,  4],
                [18,  1,  6,  5],
                [20,  2,  6,  5],
                [ 3,  3,  6,  5],
                [ 3,  4,  6,  5],
                [ 0,  5,  6,  5],
                [10,  6,  6,  5],
                [ 4,  1, 12,  6],
                [ 3,  2, 12,  6],
                [18,  3, 12,  6],
                [18,  4, 12,  6],
                [ 5,  5, 12,  6],
                [ 0,  6, 12,  6],
                [ 3,  7, 12,  6],
                [16,  8, 12,  6],
                [15,  9, 12,  6],
                [16, 10, 12,  6],
                [ 0, 11, 12,  6],
                [ 7, 12, 12,  6]
            ],
        ]
    )


@pytest.fixture
def expected_double_view_positives_mask():
    return torch.tensor(
        [
            [False, True, False, True, True, False],
            [True, False, False, True, True, False],
            [False, False, False, False, False, True],
            [True, True, False, False, True, False],
            [True, True, False, True, False, False],
            [False, False, True, False, False, False],
        ]
    )


@pytest.fixture
def expected_masked_tcrs():
    return torch.tensor(
        [
            [
                [ 2,  0,  0,  0],
                [19,  1,  6,  1],
                [18,  2,  6,  1],
                [ 8,  3,  6,  1],
                [ 7,  4,  6,  1],
                [22,  5,  6,  1],
                [ 8,  6,  6,  1],
                [14,  1,  6,  2],
                [ 3,  2,  6,  2],
                [12,  3,  6,  2],
                [ 5,  4,  6,  2],
                [ 8,  5,  6,  2],
                [12,  6,  6,  2],
                [ 1,  1, 11,  3],
                [ 1,  2, 11,  3],
                [20,  3, 11,  3],
                [11,  4, 11,  3],
                [ 3,  5, 11,  3],
                [18,  6, 11,  3],
                [ 8,  7, 11,  3],
                [18,  8, 11,  3],
                [17,  9, 11,  3],
                [12, 10, 11,  3],
                [19, 11, 11,  3],
                [18,  1,  5,  4],
                [14,  2,  5,  4],
                [ 9,  3,  5,  4],
                [12,  4,  5,  4],
                [22,  5,  5,  4],
                [ 7,  1,  6,  5],
                [ 1,  2,  6,  5],
                [14,  3,  6,  5],
                [14,  4,  6,  5],
                [ 6,  5,  6,  5],
                [10,  6,  6,  5],
                [ 4,  1, 12,  6],
                [ 1,  2, 12,  6],
                [18,  3, 12,  6],
                [18,  4, 12,  6],
                [ 5,  5, 12,  6],
                [17,  6, 12,  6],
                [ 3,  7, 12,  6],
                [16,  8, 12,  6],
                [ 1,  9, 12,  6],
                [16, 10, 12,  6],
                [ 9, 11, 12,  6],
                [ 7, 12, 12,  6],
                [ 0,  0,  0,  0]
            ],
            [
                [ 2,  0,  0,  0],
                [19,  1,  6,  1],
                [18,  2,  6,  1],
                [ 1,  3,  6,  1],
                [ 7,  4,  6,  1],
                [22,  5,  6,  1],
                [ 8,  6,  6,  1],
                [14,  1,  6,  2],
                [ 3,  2,  6,  2],
                [ 1,  3,  6,  2],
                [ 5,  4,  6,  2],
                [ 8,  5,  6,  2],
                [ 1,  6,  6,  2],
                [ 4,  1,  9,  3],
                [12,  2,  9,  3],
                [ 3,  3,  9,  3],
                [14,  4,  9,  3],
                [ 8,  5,  9,  3],
                [18,  6,  9,  3],
                [17,  7,  9,  3],
                [12,  8,  9,  3],
                [19,  9,  9,  3],
                [18,  1,  5,  4],
                [14,  2,  5,  4],
                [ 9,  3,  5,  4],
                [12,  4,  5,  4],
                [22,  5,  5,  4],
                [ 7,  1,  6,  5],
                [22,  2,  6,  5],
                [14,  3,  6,  5],
                [14,  4,  6,  5],
                [ 6,  5,  6,  5],
                [10,  6,  6,  5],
                [ 4,  1, 10,  6],
                [ 3,  2, 10,  6],
                [18,  3, 10,  6],
                [ 9,  4, 10,  6],
                [ 3,  5, 10,  6],
                [ 1,  6, 10,  6],
                [ 5,  7, 10,  6],
                [17,  8, 10,  6],
                [ 3,  9, 10,  6],
                [ 7, 10, 10,  6],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0],
                [ 0,  0,  0,  0]
            ],
            [
                [ 2,  0,  0,  0],
                [ 1,  1,  6,  1],
                [18,  2,  6,  1],
                [18,  3,  6,  1],
                [18,  4,  6,  1],
                [19,  5,  6,  1],
                [22,  6,  6,  1],
                [10,  1,  7,  2],
                [ 7,  2,  7,  2],
                [18,  3,  7,  2],
                [14,  4,  7,  2],
                [13,  5,  7,  2],
                [ 5,  6,  7,  2],
                [13,  7,  7,  2],
                [19,  1, 11,  3],
                [ 3,  2, 11,  3],
                [20,  3, 11,  3],
                [11,  4, 11,  3],
                [ 3,  5, 11,  3],
                [18,  6, 11,  3],
                [ 8,  7, 11,  3],
                [18,  8, 11,  3],
                [17,  9, 11,  3],
                [12, 10, 11,  3],
                [ 1, 11, 11,  3],
                [13,  1,  5,  4],
                [14,  2,  5,  4],
                [ 9,  3,  5,  4],
                [ 8,  4,  5,  4],
                [22,  5,  5,  4],
                [ 1,  1,  6,  5],
                [20,  2,  6,  5],
                [10,  3,  6,  5],
                [ 3,  4,  6,  5],
                [ 8,  5,  6,  5],
                [10,  6,  6,  5],
                [ 4,  1, 12,  6],
                [ 3,  2, 12,  6],
                [18,  3, 12,  6],
                [18,  4, 12,  6],
                [ 5,  5, 12,  6],
                [17,  6, 12,  6],
                [ 3,  7, 12,  6],
                [16,  8, 12,  6],
                [ 1,  9, 12,  6],
                [16, 10, 12,  6],
                [ 1, 11, 12,  6],
                [14, 12, 12,  6]
            ],
        ]
    )


@pytest.fixture
def expected_mlm_targets():
    return torch.tensor(
        [
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  3,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  9,  0, 22,  0, 22,  0,  0,  0,  0,  0,
            3,  0,  0,  0,  0,  0,  0, 15,  0,  0,  0,  0],
            [ 0,  0,  0,  8,  0,  0,  0,  0,  0, 12,  0,  0, 12,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 14,  0,  6,  0,  0,  0,  0,
            18,  0, 14,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,
            0,  0,  0,  0,  0,  0, 19,  0,  0,  0,  0,  0, 18,  0,  3,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0, 15,  0,  9,  7]
        ]
    )
