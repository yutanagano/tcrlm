import pytest
import torch
from source.utils.nn import create_padding_mask


# Positive tests
def test_create_padding_mask():
    x = torch.tensor(
        [[1,2,3,21,21]],
        dtype=torch.long
    )
    expected = torch.tensor(
        [[False,False,False,True,True]],
        dtype=torch.bool
    )
    padding_mask = create_padding_mask(x)

    assert(torch.equal(padding_mask,expected))