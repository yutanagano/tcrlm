'''
CDR3BERT classes
'''


from .embedder import MLMEmbedder
import math
from src.utils import masked_average_pool
import torch
from torch import Tensor
from torch.nn import Embedding, Module
from torch.nn.functional import normalize
from typing import Tuple


class AAEmbedding_c(Module):
    '''
    CDR3 embedder with only chain information
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


class CDR3BERT_c(MLMEmbedder):
    '''
    CDR3BERT model that only gets amino acid and chain information.

    Compatible tokenisers: CDR3Tokeniser
    '''

    
    def __init__(
        self,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_layer = num_encoder_layers - 1

        self._num_layers = num_encoder_layers
        self._d_model = d_model
        self._nhead = nhead
        self._dim_feedforward = dim_feedforward

        # Create an instance of the encoder layer that we want
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Create a stack of num_layers * encoder layer, our main network
        self.encoder_stack = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Create an embedder that can take in a LongTensor representing padded
        # batch of cdr3 sequences, and output a similar FloatTensor with an
        # extra dimension representing the embedding dimension.
        self.embedder = AAEmbedding_c(
            embedding_dim=d_model
        )

        # Create a linear layer that can project the encoder stack's token
        # embedding outputs to 20-dimensional vectors that can represent a
        # probability distribution over all amino acids (for MLM)
        self.generator = torch.nn.Linear(
            in_features=d_model,
            out_features=20
        )


    @property
    def name(self) -> str:
        return f'CDR3BERT_c_{self._num_layers}_{self._d_model}_'\
            f'{self._nhead}_{self._dim_feedforward}-embed_{self.embed_layer}'


    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        padding_mask = (x[:,:,0] == 0)

        # Create an embedding of the input tensor (with positional info)
        x_emb = self.embedder(x)

        # Run the embedded input through the bert stack
        out = self.encoder_stack(
            src=x_emb,
            src_key_padding_mask=padding_mask
        )

        return out, padding_mask


    def embed(self, x: Tensor) -> Tensor:
        # Run the input partially through the BERT stack
        padding_mask = (x[:,:,0] == 0)
        x_emb = self.embedder(x)
        for layer in self.encoder_stack.layers[:self.embed_layer]:
            x_emb = layer(src=x_emb, src_key_padding_mask=padding_mask)

        # Compute the masked average pool
        x_emb = masked_average_pool(x_emb, padding_mask)

        # l2 norm and return
        return normalize(x_emb, p=2, dim=1)


    def mlm(self, x: Tensor) -> Tensor:
        return self.generator(self.forward(x)[0])