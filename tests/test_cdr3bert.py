import pytest
import torch
from source.cdr3bert import Cdr3Bert


@pytest.fixture(scope='module')
def instantiate_bert():
    bert = Cdr3Bert(num_encoder_layers=2,
                    d_model=6,
                    nhead=2,
                    dim_feedforward=48)
    yield bert


# Positive tests
def test_cdr3bert(instantiate_bert):
    bert = instantiate_bert
    batch = torch.zeros((3,6), dtype=torch.int)
    mask = torch.zeros((3,6), dtype=torch.bool)
    out = bert(x=batch,
               padding_mask=mask)

    assert(out.size() == (3,6,20))