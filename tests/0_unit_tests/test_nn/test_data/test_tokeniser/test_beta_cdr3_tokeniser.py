import pytest
import torch

from src.nn.data import schema
from src.nn.data.tokeniser import BetaCdr3Tokeniser


def test_tokenise(tokeniser, mock_tcr):
    tokenised_tcr = tokeniser.tokenise(mock_tcr)
    expected = torch.tensor(
        [
            [2, 0, 0],
            [4, 1, 6],
            [3, 2, 6],
            [18, 3, 6],
            [16, 4, 6],
            [22, 5, 6],
            [7, 6, 6],
        ]
    )

    assert torch.equal(tokenised_tcr, expected)


def test_tokenise_tcr_with_empty_beta_junction(tokeniser):
    tcr_with_empty_beta = schema.make_tcr_from_components("TRAV1-1*01", "CASQYF", "TRBV2*01", None)

    with pytest.raises(RuntimeError):
        tokeniser.tokenise(tcr_with_empty_beta)


@pytest.fixture
def tokeniser():
    return BetaCdr3Tokeniser()