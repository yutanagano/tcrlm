'''
Various amino acid embedding modules.
'''


import math
from src.modules.bert.embedding.sinpos import SinPositionEmbedding
from torch import Tensor
from torch.nn import Embedding, Module


class CDR3Embedding_a(Module):
    '''
    CDR3 embedder which encodes amino acid information information only.

    Compatible tokenisers: CDR3Tokeniser
    '''
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.token_embedding = Embedding(
            num_embeddings=23, # <pad> + <mask> + <cls> + 20 amino acids
            embedding_dim=embedding_dim,
            padding_idx=0
        )
    

    def forward(self, x: Tensor) -> Tensor:
        return \
            self.token_embedding(x[:,:,0]) * math.sqrt(self.embedding_dim)


class CDR3Embedding_ap(CDR3Embedding_a):
    '''
    CDR3 embedder which encodes amino acid and residue position information.

    Compatible tokenisers: CDR3Tokeniser
    '''
    def __init__(self, embedding_dim: int) -> None:
        super().__init__(embedding_dim)

        self.position_embedding = SinPositionEmbedding(
            num_embeddings=100,
            embedding_dim=embedding_dim
        )
    

    def forward(self, x: Tensor) -> Tensor:
        return \
            (
                self.token_embedding(x[:,:,0]) +
                self.position_embedding(x[:,:,1])
            ) * math.sqrt(self.embedding_dim)


class CDR3Embedding_ac(CDR3Embedding_a):
    '''
    CDR3 embedder which encodes amino acid and chain information.

    Compatible tokenisers: CDR3Tokeniser
    '''
    def __init__(self, embedding_dim: int) -> None:
        super().__init__(embedding_dim)

        self.chain_embedding = Embedding(
            num_embeddings=3, # <pad>, alpha, beta
            embedding_dim=embedding_dim,
            padding_idx=0
        )
    

    def forward(self, x: Tensor) -> Tensor:
        return \
            (self.token_embedding(x[:,:,0]) + self.chain_embedding(x[:,:,2])) \
                * math.sqrt(self.embedding_dim)


class CDR3Embedding_apc(CDR3Embedding_ap):
    '''
    CDR3 embedder which encodes amino acid, chain, and residue position
    information.

    Compatible tokenisers: CDR3Tokeniser
    '''
    def __init__(self, embedding_dim: int) -> None:
        super().__init__(embedding_dim)

        self.chain_embedding = Embedding(
            num_embeddings=3, # <pad>, alpha, beta
            embedding_dim=embedding_dim,
            padding_idx=0
        )
    

    def forward(self, x: Tensor) -> Tensor:
        return \
            (
                self.token_embedding(x[:,:,0]) +
                self.position_embedding(x[:,:,1]) +
                self.chain_embedding(x[:,:,2])
            ) * math.sqrt(self.embedding_dim)