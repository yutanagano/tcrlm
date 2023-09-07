import pytest
import torch
from torch import testing

from src.nn.performance_measure import average_positive_distance, average_negative_distance


def test_average_positive_distance(mock_tcr_representations, mock_positives_mask):
    result = average_positive_distance(mock_tcr_representations, mock_positives_mask)
    expected = 0.5

    testing.assert_close(result, expected)


def test_average_negative_distance(mock_tcr_representations, mock_positives_mask):
    result = average_negative_distance(mock_tcr_representations, mock_positives_mask)
    expected = 1.08934712

    testing.assert_close(result, expected)


@pytest.fixture
def mock_tcr_representations():
    return torch.tensor(
        [
            [1,   0],
            [0,   1],
            [0.5, 0],
            [0, 0.5]
        ],
        dtype=torch.float32
    )


@pytest.fixture
def mock_positives_mask():
    return torch.tensor(
        [
            [0,0,1,0],
            [0,0,0,1],
            [1,0,0,0],
            [0,1,0,0]
        ],
        dtype=torch.bool
    )