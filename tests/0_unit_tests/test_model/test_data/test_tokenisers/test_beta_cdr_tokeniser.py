import pytest
import torch

from src.tcr import TravGene, TrbvGene, Tcrv, Tcr
from src.model.data.tokenisers.beta_cdr_tokeniser import BetaCdrTokeniser


def test_tokenise(mock_tcr: Tcr):
    tokeniser = BetaCdrTokeniser()

    tokenised_tcr = tokeniser.tokenise(mock_tcr)
    expected = torch.tensor(
        [
            [2,0,0,0],

            [18,1,5,1],
            [14,2,5,1],
            [9,3,5,1],
            [12,4,5,1],
            [22,5,5,1],

            [7,1,6,2],
            [22,2,6,2],
            [14,3,6,2],
            [14,4,6,2],
            [6,5,6,2],
            [10,6,6,2],

            [4,1,6,3],
            [3,2,6,3],
            [18,3,6,3],
            [16,4,6,3],
            [22,5,6,3],
            [7,6,6,3]
        ]
    )

    assert torch.equal(tokenised_tcr, expected)