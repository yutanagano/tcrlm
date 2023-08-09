import pytest
import torch
from torch import Tensor

from src.data.tokeniser.beta_cdr_tokeniser import BetaCdrTokeniser
from src.model.token_embedder.beta_cdr_embedder import BetaCdrEmbedder


@pytest.fixture
def mock_batch_of_tokenised_tcrs(mock_tcr):
    tokeniser = BetaCdrTokeniser()
    mock_tokenised_tcr = tokeniser.tokenise(mock_tcr)
    mock_batch_of_tokenised_tcrs = torch.stack([mock_tokenised_tcr, mock_tokenised_tcr])

    return mock_batch_of_tokenised_tcrs

def test_forward(mock_batch_of_tokenised_tcrs):
    BATCH_SIZE = 2
    TOKENISED_TCR_LENGTH = 18
    EMBEDDING_DIM = 10

    embedder = BetaCdrEmbedder(embedding_dim=EMBEDDING_DIM)
    mock_batch_of_embedded_tcrs = embedder.forward(mock_batch_of_tokenised_tcrs)

    assert type(mock_batch_of_embedded_tcrs) == Tensor
    assert mock_batch_of_embedded_tcrs.dim() == 3
    assert mock_batch_of_embedded_tcrs.size(0) == BATCH_SIZE
    assert mock_batch_of_embedded_tcrs.size(1) == TOKENISED_TCR_LENGTH
    assert mock_batch_of_embedded_tcrs.size(2) == EMBEDDING_DIM