'Helper functions and classes for the custom nn models.'


import math
import torch
import torch.nn as nn


def create_padding_mask(x: torch.Tensor) -> torch.Tensor:
    return x == 21


def masked_average_pool(
    token_embeddings: torch.Tensor,
    padding_mask: torch.Tensor
) -> torch.Tensor:
    '''
    Take sequences of token embeddings produced by the Cdr3Bert model, as well
    the corresponding token padding mask tensor, and generate fixed-size
    vector embeddings for each token sequence by averaging all token embeddings
    across each sequence.
    Input:
    1. Batched sequences of token embeddings                (size: N,S,E)*
    2. Padding mask                                         (size: N,S)*
    Output: Batched vector embeddings of the cdr3 sequences (size: N,E)*

    * Dimensions are as follows:
    N - number of items in batch i.e. batch size
    S - number of tokens in sequence i.e. sequence length
    V - vocabulary size (in this case 20 for 20 amino acids)
    '''
    # Reverse the boolean values of the mask to mark where the tokens are, as
    # opposed to where the tokens are not. Then, resize padding mask to make it
    # broadcastable with token embeddings
    padding_mask = padding_mask.logical_not().unsqueeze(-1)

    # Compute averages of token embeddings per sequence, ignoring padding tokens
    token_embeddings_masked = token_embeddings * padding_mask
    token_embeddings_summed = token_embeddings_masked.sum(1)
    token_embeddings_averaged = token_embeddings_summed / padding_mask.sum(1)

    return token_embeddings_averaged


class AaEmbedder(nn.Module):
    '''
    Helper class to convert tensor of input indices to corresponding tensor of
    token embeddings.
    '''
    def __init__(self, embedding_dim: int):
        super(AaEmbedder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=22, # 20 aa's + mask + pad
                                      embedding_dim=embedding_dim,
                                      padding_idx=21)
        self.embedding_dim = embedding_dim
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.embedding_dim)


class PositionEncoder(nn.Module):
    '''
    Helper class that adds positional encoding to the token embedding to
    infuse the embeddings with information concerning token order.
    '''
    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.1,
        max_len: int = 200
    ):
        # Ensure that the embedding has an even number of dimensions
        assert(embedding_dim % 2 == 0)

        super(PositionEncoder, self).__init__()

        # Create a tensor to store pre-calculated positional embeddings
        position_encoding = torch.zeros(max_len, embedding_dim)

        # Calculate the positional embeddings
        position_indices = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(-math.log(30) * \
                             torch.arange(0, embedding_dim, 2) / embedding_dim)
        
        position_encoding[:, 0::2] = torch.sin(position_indices * div_term)
        position_encoding[:, 1::2] = torch.cos(position_indices * div_term)

        # Add an extra dimension so that the tensor shape is now (1, max_len, 
        # embedding_dim).
        position_encoding = position_encoding.unsqueeze(0)

        # Register this tensor as a buffer
        self.register_buffer('position_encoding', position_encoding)

        # Create a dropout module
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.position_encoding[:, :x.size(1)])