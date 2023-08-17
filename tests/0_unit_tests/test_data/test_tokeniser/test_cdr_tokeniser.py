import pytest
import torch

from src.tcr import Tcr, Tcrv, TravGene, TrbvGene
from src.data.tokeniser.cdr_tokeniser import CdrTokeniser


@pytest.fixture
def tokeniser():
    return CdrTokeniser()


def test_tokenise(tokeniser: CdrTokeniser):
    example_tcr = Tcr(
        trav=Tcrv(TravGene["TRAV1-1"], 1),
        junction_a_sequence="CASQYF",
        trbv=Tcrv(TrbvGene["TRBV2"], 1),
        junction_b_sequence="CATQYF",
    )

    tokenised_tcr = tokeniser.tokenise(example_tcr)
    expected = torch.tensor(
        [
            [2, 0, 0, 0],
            [19, 1, 6, 1],
            [18, 2, 6, 1],
            [8, 3, 6, 1],
            [7, 4, 6, 1],
            [22, 5, 6, 1],
            [8, 6, 6, 1],
            [14, 1, 6, 2],
            [3, 2, 6, 2],
            [12, 3, 6, 2],
            [5, 4, 6, 2],
            [8, 5, 6, 2],
            [12, 6, 6, 2],
            [4, 1, 6, 3],
            [3, 2, 6, 3],
            [18, 3, 6, 3],
            [16, 4, 6, 3],
            [22, 5, 6, 3],
            [7, 6, 6, 3],
            [18, 1, 5, 4],
            [14, 2, 5, 4],
            [9, 3, 5, 4],
            [12, 4, 5, 4],
            [22, 5, 5, 4],
            [7, 1, 6, 5],
            [22, 2, 6, 5],
            [14, 3, 6, 5],
            [14, 4, 6, 5],
            [6, 5, 6, 5],
            [10, 6, 6, 5],
            [4, 1, 6, 6],
            [3, 2, 6, 6],
            [19, 3, 6, 6],
            [16, 4, 6, 6],
            [22, 5, 6, 6],
            [7, 6, 6, 6],
        ]
    )

    assert torch.equal(tokenised_tcr, expected)
