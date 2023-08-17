import pytest
import torch
from torch import Tensor

from src.model.bert import Bert
from src.model.token_embedder.beta_cdr_embedder import BetaCdrEmbedder
from src.model.self_attention_stack import SelfAttentionStackWithBuiltins
from src.model.mlm_token_prediction_projector import AminoAcidTokenProjector
from src.model.vector_representation_delegate import (
    AveragePoolVectorRepresentationDelegate,
)


D_MODEL = 4

BATCH_SIZE = 5
TOKENISED_TCR_LENGTH = 5
TOKEN_DIMENSIONALITY = 4

AMINO_ACID_VOCABULARY_SIZE = 20


@pytest.fixture
def bert():
    token_embedder = BetaCdrEmbedder(embedding_dim=D_MODEL)
    self_attention_stack = SelfAttentionStackWithBuiltins(
        num_layers=2, d_model=D_MODEL, nhead=2
    )
    mlm_token_prediction_projector = AminoAcidTokenProjector(d_model=D_MODEL)
    vector_representation_delegate = AveragePoolVectorRepresentationDelegate(
        self_attention_stack=self_attention_stack
    )

    bert = Bert(
        token_embedder=token_embedder,
        self_attention_stack=self_attention_stack,
        mlm_token_prediction_projector=mlm_token_prediction_projector,
        vector_representation_delegate=vector_representation_delegate,
    )

    return bert


@pytest.fixture
def mock_tokenised_tcrs():
    return torch.ones(
        (BATCH_SIZE, TOKENISED_TCR_LENGTH, TOKEN_DIMENSIONALITY), dtype=torch.long
    )


def test_get_vector_representations_of(bert: Bert, mock_tokenised_tcrs):
    result = bert.get_vector_representations_of(mock_tokenised_tcrs)

    assert type(result) == Tensor
    assert result.dim() == 2
    assert result.size(0) == BATCH_SIZE
    assert result.size(1) == D_MODEL


def test_get_mlm_token_predictions_for(bert: Bert, mock_tokenised_tcrs):
    result = bert.get_mlm_token_predictions_for(mock_tokenised_tcrs)

    assert type(result) == Tensor
    assert result.dim() == 3
    assert result.size(0) == BATCH_SIZE
    assert result.size(1) == TOKENISED_TCR_LENGTH
    assert result.size(2) == AMINO_ACID_VOCABULARY_SIZE
