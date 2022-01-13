import pytest
import torch
from source.data_handling import CDR3Tokeniser


@pytest.fixture(scope='module')
def instantiate_tokeniser():
    tokeniser = CDR3Tokeniser()
    yield tokeniser


# Positive tests
def test_tokenise(instantiate_tokeniser):
    tokeniser = instantiate_tokeniser
    tokenised = tokeniser.tokenise('?ACDEF-')
    expected = torch.tensor([0,1,2,3,4,5,21], dtype=torch.int)

    assert(torch.equal(tokenised, expected))


def test_to_string(instantiate_tokeniser):
    tokeniser = instantiate_tokeniser
    tokenised = torch.tensor([0,1,2,3,4,5,21], dtype=torch.int)
    stringified = tokeniser.to_string(tokenised)
    expected = '?ACDEF-'

    assert(stringified == expected)