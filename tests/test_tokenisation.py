import pytest
import torch
from source.data_handling.utils import tokenise


# Positive tests
def test_tokenise():
    tokenised = tokenise(['?','A','C','D','E','F','-'])
    expected = torch.tensor([20,0,1,2,3,4,21], dtype=torch.long)

    assert(torch.equal(tokenised, expected))
