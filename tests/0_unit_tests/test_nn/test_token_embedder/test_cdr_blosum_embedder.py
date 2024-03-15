import pytest
import torch
from torch import Tensor
from src.nn.data.tokeniser import CdrTokeniser
from src.nn.token_embedder import CdrBlosumEmbedder


def test_forward(mock_batch_of_tokenised_tcrs):
    BATCH_DIMENSIONALITY = 3
    BATCH_SIZE = 2
    TOKENISED_TCR_LENGTH = 36
    EMBEDDING_DIM = 20+1+6

    embedder = CdrBlosumEmbedder()
    mock_batch_of_embedded_tcrs = embedder.forward(mock_batch_of_tokenised_tcrs)

    assert type(mock_batch_of_embedded_tcrs) == Tensor

    assert mock_batch_of_embedded_tcrs.dim() == BATCH_DIMENSIONALITY
    assert mock_batch_of_embedded_tcrs.size(0) == BATCH_SIZE
    assert mock_batch_of_embedded_tcrs.size(1) == TOKENISED_TCR_LENGTH
    assert mock_batch_of_embedded_tcrs.size(2) == EMBEDDING_DIM


@pytest.fixture
def mock_batch_of_tokenised_tcrs(mock_tcr):
    tokeniser = CdrTokeniser()
    tokenised_mock_tcr = tokeniser.tokenise(mock_tcr)
    batch_of_tokenised_mock_tcrs = torch.stack([tokenised_mock_tcr, tokenised_mock_tcr])

    return batch_of_tokenised_mock_tcrs
