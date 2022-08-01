import pytest
import torch
from source.utils.nn import AaEmbedder


@pytest.fixture(scope='module')
def instantiate_embedder():
    embedder = AaEmbedder(embedding_dim=6)
    yield embedder


# Positive tests
def test_embedding(instantiate_embedder):
    embedder = instantiate_embedder
    batch = torch.tensor(
        [
            [1,2,3,21,21,21],
            [1,2,21,21,21,21],
            [1,0,3,0,5,21]
        ],
        dtype=torch.int
    )
    embedded_batch = embedder(batch)

    assert(embedded_batch.size() == (3,6,6))