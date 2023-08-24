from enum import IntEnum
import torch
from torch import FloatTensor, LongTensor
from torch.nn import Module

from src.data.tokeniser.token_indices import (
    DefaultTokenIndex,
    AminoAcidTokenIndex,
    BetaCdrCompartmentIndex,
)
from src.model.token_embedder.token_embedder import TokenEmbedder


class BetaCdrSimpleEmbedder(TokenEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self._token_embedding = OneHotTokenIndexEmbedding(AminoAcidTokenIndex)
        self._position_embedding = SimpleRelativePositionEmbedding()
        self._compartment_embedding = OneHotTokenIndexEmbedding(BetaCdrCompartmentIndex)

    def forward(self, tokenised_tcrs: LongTensor) -> FloatTensor:
        token_component = self._token_embedding.forward(tokenised_tcrs[:, :, 0])
        position_component = self._position_embedding.forward(tokenised_tcrs[:, :, 1:3])
        compartment_component = self._compartment_embedding.forward(tokenised_tcrs[:, :, 3])
        all_components_stacked = torch.concatenate([token_component, position_component, compartment_component], dim=-1)
        return all_components_stacked


class OneHotTokenIndexEmbedding(Module):
    def __init__(self, token_index: IntEnum) -> None:
        super().__init__()
        self._register_token_embeddings(token_index)

    def _register_token_embeddings(self, token_index: IntEnum) -> FloatTensor:
        num_tokens = len(token_index)
        num_tokens_excluding_null = num_tokens - 1

        null_embedding = torch.zeros((1, num_tokens_excluding_null))
        non_null_token_embeddings = torch.eye(num_tokens_excluding_null)
        token_embeddings = torch.concatenate([null_embedding, non_null_token_embeddings], dim=0)

        self.register_buffer("_token_embeddings", token_embeddings)
    
    def forward(self, token_indices: LongTensor) -> FloatTensor:
        return self._token_embeddings[token_indices]


class SimpleRelativePositionEmbedding(Module):
    def forward(self, position_indices: LongTensor) -> FloatTensor:
        """
        Input tensor should have shape (..., 2) with first of the final two dimensions encoding token position, and the final dimension encoding compartment length.
        """
        null_mask = position_indices[...,0] == DefaultTokenIndex.NULL
        zero_indexed_token_positions = position_indices[...,0] - 1
        compartment_length_minus_one = position_indices[...,1] - 1
        
        relative_token_positions = zero_indexed_token_positions / compartment_length_minus_one

        RELATIVE_POSITION_IF_ONLY_ONE_TOKEN_IN_COMPARTMENT = 0.5
        relative_token_positions[relative_token_positions.isnan()] = RELATIVE_POSITION_IF_ONLY_ONE_TOKEN_IN_COMPARTMENT
        relative_token_positions[null_mask] = 0

        relative_token_positions = relative_token_positions.unsqueeze(dim=-1)

        return relative_token_positions