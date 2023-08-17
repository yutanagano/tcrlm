from torch import Tensor
from torch.nn import Module

from src.data.tokeniser.token_indices import DefaultTokenIndex
from src.model.token_embedder.token_embedder import TokenEmbedder
from src.model.mlm_token_prediction_projector import MlmTokenPredictionProjector
from src.model.self_attention_stack import SelfAttentionStack
from src.model.vector_representation_delegate import VectorRepresentationDelegate


class Bert(Module):
    def __init__(
        self,
        token_embedder: TokenEmbedder,
        self_attention_stack: SelfAttentionStack,
        mlm_token_prediction_projector: MlmTokenPredictionProjector,
        vector_representation_delegate: VectorRepresentationDelegate,
    ) -> None:
        super().__init__()

        self._token_embedder = token_embedder
        self._self_attention_stack = self_attention_stack
        self._mlm_token_prediction_projector = mlm_token_prediction_projector
        self._vector_representation_delegate = vector_representation_delegate

    @property
    def d_model(self) -> int:
        return self._self_attention_stack.d_model

    def get_vector_representations_of(self, tokenised_tcrs: Tensor) -> Tensor:
        raw_token_embeddings = self._embed(tokenised_tcrs)
        padding_mask = self._get_padding_mask(tokenised_tcrs)
        vector_representations = (
            self._vector_representation_delegate.get_vector_representations_of(
                raw_token_embeddings, padding_mask
            )
        )

        return vector_representations

    def get_mlm_token_predictions_for(
        self, tokenised_and_masked_tcrs: Tensor
    ) -> Tensor:
        raw_token_embeddings = self._embed(tokenised_and_masked_tcrs)
        padding_mask = self._get_padding_mask(tokenised_and_masked_tcrs)
        contextualised_token_embeddings = self._self_attention_stack.forward(
            raw_token_embeddings, padding_mask
        )
        mlm_token_predictions = self._mlm_token_prediction_projector.forward(
            contextualised_token_embeddings
        )

        return mlm_token_predictions

    def _embed(self, tokenised_tcrs: Tensor) -> Tensor:
        return self._token_embedder.forward(tokenised_tcrs)

    def _get_padding_mask(self, tokenised_tcrs: Tensor) -> Tensor:
        return tokenised_tcrs[:, :, 0] == DefaultTokenIndex.NULL
