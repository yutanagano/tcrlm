import pytest
import torch
from source.data_handling import CDR3Tokeniser


@pytest.fixture(scope='module')
def instantiate_tokeniser():
    tokeniser = CDR3Tokeniser()
    yield tokeniser


# Positive tests
def test_tokenise_in(instantiate_tokeniser):
    tokeniser = instantiate_tokeniser
    tokenised = tokeniser.tokenise_in('?ACDEF-')
    expected = torch.tensor([0,1,2,3,4,5,21], dtype=torch.long)

    assert(torch.equal(tokenised, expected))


def test_tokenise_out(instantiate_tokeniser):
    tokeniser = instantiate_tokeniser
    tokenised = tokeniser.tokenise_out('ACDEF-')
    expected = torch.tensor([0,1,2,3,4,21], dtype=torch.long)

    assert(torch.equal(tokenised, expected))