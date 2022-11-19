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
    ):
        assert embedding_dim % 2 == 0
        self._embedding_dim = embedding_dim

        super().__init__()

        position_embedding = torch.zeros(num_embeddings+1, embedding_dim)
        position_indices = torch.arange(0, num_embeddings).unsqueeze(1)
        div_term = torch.exp(-math.log(sin_scale_factor) * \
                             torch.arange(0, embedding_dim, 2) / embedding_dim)
        position_embedding[1:, 0::2] = torch.sin(position_indices * div_term)
        position_embedding[1:, 1::2] = torch.cos(position_indices * div_term)

        self.register_buffer('position_embedding', position_embedding)


    def forward(self, x: int) -> Tensor:
        return self.position_embedding[x-1]


class AAEmbedding_c(Module):
    '''
    CDR3 embedder which encodes amino acid and chain information.
    '''
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.token_embedding = Embedding(
            num_embeddings=23, # <pad> + <mask> + <cls> + 20 amino acids
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.chain_embedding = Embedding(
            num_embeddings=3, # <pad>, alpha, beta
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.embedding_dim = embedding_dim
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return \
            (self.token_embedding(x[:,:,0]) + self.chain_embedding(x[:,:,1])) \
                * math.sqrt(self.embedding_dim)


class AAEmbedding_cp(Module):
    '''
    CDR3 embedder which encodes amino acid, chain, and residue position
    information.
    '''
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.token_embedding = Embedding(
            num_embeddings=23, # <pad> + <mask> + <cls> + 20 amino acids
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.chain_embedding = Embedding(
            num_embeddings=3, # <pad>, alpha, beta
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.position_embedding = SinPositionEmbedding(
            num_embeddings=100,
            embedding_dim=embedding_dim
        )
        self.embedding_dim = embedding_dim
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return \
            (
                self.token_embedding(x[:,:,0]) +
                self.chain_embedding(x[:,:,1]) +
                self.position_embedding(x[:,:,2])
            ) * math.sqrt(self.embedding_dim)