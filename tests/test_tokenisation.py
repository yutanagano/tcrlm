import pytest
import torch
from source.data_handling import tokenise, lookup


# Positive tests
def test_tokenise():
    tokenised = tokenise(['?','A','C','D','E','F','-'])
    expected = torch.tensor([20,0,1,2,3,4,21], dtype=torch.long)

    assert(torch.equal(tokenised, expected))


def test_lookup():
    expected = ['A','C','D','E','F']
    for i in range(5):
        amino_acid = lookup(i)
        assert(amino_acid == expected[i])
