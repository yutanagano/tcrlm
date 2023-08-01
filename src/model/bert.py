from abc import abstractmethod, ABC
from torch import Tensor
from torch.nn import Module

from src.model.data.tokenisers.token_indices import DefaultTokenInex
from src.model.token_embedders.token_embedder import TokenEmbedder
from src.model.mlm_token_prediction_projector import MlmTokenPredictionProjector
from src.model.self_attention_stack import SelfAttentionStack
from src.model.vector_representation_delegate import VectorRepresentationDelegate


class Bert(ABC, Module):
    @property
    @abstractmethod
    def embedder(self) -> TokenEmbedder:
        pass

    @property
    @abstractmethod
    def self_attention_stack(self) -> SelfAttentionStack:
        pass

    @property
    @abstractmethod
    def mlm_token_prediction_projector(self) -> MlmTokenPredictionProjector:
        pass

    @property
    @abstractmethod
    def vector_representation_delegate(self) -> VectorRepresentationDelegate:
        pass

    def get_vector_representations_of(self, tokenised_tcrs: Tensor) -> Tensor:
        raw_token_embeddings = self.embedder.forward(tokenised_tcrs)
        padding_mask = self._get_padding_mask(tokenised_tcrs)
        return self.vector_representation_delegate.get_vector_representations_of(raw_token_embeddings, padding_mask)

    def get_mlm_token_predictions(self, tokenised_and_masked_tcrs: Tensor) -> Tensor:
        raw_token_embeddings = self.embedder.forward(tokenised_and_masked_tcrs)
        padding_mask = self._get_padding_mask(tokenised_and_masked_tcrs)
        contextualised_token_embeddings = self.self_attention_stack.forward(raw_token_embeddings, padding_mask)
        mlm_token_predictions = self.mlm_token_prediction_projector.forward(contextualised_token_embeddings)

        return mlm_token_predictions

    def _get_padding_mask(self, tokenised_tcrs: Tensor) -> Tensor:
        return tokenised_tcrs[:, :, 0] == DefaultTokenInex.NULL