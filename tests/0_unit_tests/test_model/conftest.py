import pytest

from src.model.bert import Bert
from src.model.token_embedder.beta_cdr_embedder import BetaCdrEmbedder
from src.model.self_attention_stack import SelfAttentionStackWithBuiltins
from src.model.mlm_token_prediction_projector import AminoAcidTokenProjector
from src.model.vector_representation_delegate import (
    AveragePoolVectorRepresentationDelegate,
)


@pytest.fixture
def toy_bert(toy_bert_d_model):
    token_embedder = BetaCdrEmbedder(embedding_dim=toy_bert_d_model)
    self_attention_stack = SelfAttentionStackWithBuiltins(
        num_layers=2, d_model=toy_bert_d_model, nhead=2
    )
    mlm_token_prediction_projector = AminoAcidTokenProjector(d_model=toy_bert_d_model)
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
def toy_bert_d_model():
    return 4