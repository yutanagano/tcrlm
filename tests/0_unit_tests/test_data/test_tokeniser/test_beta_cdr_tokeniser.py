import pytest
import torch

from src.tcr import Tcr, Tcrv, TravGene, TrbvGene
from src.data.tokeniser.beta_cdr_tokeniser import BetaCdrTokeniser


@pytest.fixture
def tokeniser():
    return BetaCdrTokeniser()

def test_tokenise(tokeniser: BetaCdrTokeniser):
    example_tcr = Tcr(
        trav=Tcrv(TravGene["TRAV1-1"], 1),
        junction_a_sequence="CASQYF",
        trbv=Tcrv(TrbvGene["TRBV2"], 1),
        junction_b_sequence="CATQYF"
    )

    tokenised_tcr = tokeniser.tokenise(example_tcr)
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
            [19,3,6,3],
            [16,4,6,3],
            [22,5,6,3],
            [7,6,6,3]
        ]
    )

    assert torch.equal(tokenised_tcr, expected)

def test_tokenise_tcr_with_empty_beta(tokeniser: BetaCdrTokeniser):
    tcr_with_empty_beta = Tcr(
        trav=Tcrv(TravGene["TRAV1-1"], 1),
        junction_a_sequence="CASQYF",
        trbv=Tcrv(None, None),
        junction_b_sequence=None
    )

    with pytest.raises(RuntimeError):
        tokeniser.tokenise(tcr_with_empty_beta)