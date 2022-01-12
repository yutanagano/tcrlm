import pytest
import torch
from source.cdr3bert import PositionEncoder


@pytest.fixture(scope='module')
def instantiate_position_encoder():
    position_encoder = PositionEncoder(embedding_dim=6)
    yield position_encoder


# Positive tests
def test_position_encoder(instantiate_position_encoder):
    encoder = instantiate_position_encoder
    batch = torch.zeros((3,6,6), dtype=torch.float)
    batch = encoder(batch)

    assert(batch.size() == (3,6,6))