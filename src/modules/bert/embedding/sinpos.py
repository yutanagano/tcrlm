'''
Modules to embed various TCR tokens.
'''


import math
import torch
from torch import Tensor
from torch.nn import Embedding, Module


class SinPositionEmbedding(Module):
    '''
    Module to encode positional embeddings via a stacked sinusoidal function.
    '''
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        sin_scale_factor: int = 30
    ) -> None:
        assert embedding_dim % 2 == 0
        self._embedding_dim = embedding_dim

        super().__init__()

        # 0th dim size is num_embeddings+1 to account for fact that 0 is null value (positions are 1-indexed)
        position_embedding = torch.zeros(num_embeddings+1, embedding_dim)
        position_indices = torch.arange(0, num_embeddings).unsqueeze(1)
        div_term = torch.exp(-math.log(sin_scale_factor) * \
                             torch.arange(0, embedding_dim, 2) / embedding_dim)
        position_embedding[1:, 0::2] = torch.sin(position_indices * div_term)
        position_embedding[1:, 1::2] = torch.cos(position_indices * div_term)

        self.register_buffer('position_embedding', position_embedding)


    def forward(self, x: Tensor) -> Tensor:
        return self.position_embedding[x]