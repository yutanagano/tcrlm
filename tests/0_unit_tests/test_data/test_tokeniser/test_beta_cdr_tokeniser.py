import pytest
import torch

from src.data import make_tcr_from_components
from src.data.tokeniser import BetaCdrTokeniser


def test_tokenise(tokeniser: BetaCdrTokeniser, mock_tcr):
    tokenised_tcr = tokeniser.tokenise(mock_tcr)
    expected = torch.tensor(
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
            [4, 1, 6, 3],
            [3, 2, 6, 3],
            [18, 3, 6, 3],
            [16, 4, 6, 3],
            [22, 5, 6, 3],
            [7, 6, 6, 3],
        ]
    )

    assert torch.equal(tokenised_tcr, expected)


def test_tokenise_tcr_with_empty_beta(tokeniser: BetaCdrTokeniser):
    tcr_with_empty_beta = make_tcr_from_components("TRAV1-1*01", "CASQYF", None, None)

    with pytest.raises(RuntimeError):
        tokeniser.tokenise(tcr_with_empty_beta)


@pytest.fixture
def tokeniser():
    return BetaCdrTokeniser()