'''
cdr3bert.py
purpose: Python module with classes that represent the code base for the BERT-
         based neural network models that will be able to learn and process TCR
         beta-chain CDR3 sequences.
author: Yuta Nagano
ver: 1.0.0
'''


import math
import torch
from torch import nn


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
    
    def forward(self, x: torch.Tensor):
        return self.embedding(x) * math.sqrt(self.embedding_dim)


class PositionEncoder(nn.Module):
    '''
    Helper class that adds positional encoding to the token embedding to
    infuse the embeddings with information concerning token order.
    '''
    def __init__(self,
                 embedding_dim: int,
                 dropout: float = 0.1,
                 max_len: int = 200):
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

    def forward(self, x: torch.Tensor):
        return self.dropout(x + self.position_encoding[:, :x.size(1)])


class Cdr3Bert(nn.Module):
    '''
    Neural network based on the BERT architecture modified to process TCR beta-
    chain CDR3 sequences.
    '''
    def __init__(self,
                 num_encoder_layers: int,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 layer_norm_eps: float = 1e-5):
        
        super(Cdr3Bert, self).__init__()

        # Create an instance of the encoder layer that we want
        encoder_layer = nn.TransformerEncoderLayer(
                            d_model=d_model,
                            nhead=nhead,
                            dim_feedforward=dim_feedforward,
                            dropout=dropout,
                            activation=activation,
                            layer_norm_eps=layer_norm_eps,
                            batch_first=True)
        
        # Create a stack of num_layers * encoder layer, this is our main network
        self.encoder_stack = nn.TransformerEncoder(
                                encoder_layer=encoder_layer,
                                num_layers=num_encoder_layers)
        
        # Create a fully-connected layer that will function as the final layer,
        # which projects the fully processed token embeddings onto a probability
        # distribution over all possible amino acid residues.
        self.generator = nn.Linear(in_features=d_model,
                                   out_features=20)
        
        # Create an embedder that can take in a LongTensor representing a padded
        # batch of cdr3 sequences, and output a similar FloatTensor with an
        # extra dimension representing the embedding dimension.
        self.embedder = AaEmbedder(embedding_dim=d_model)

        # Create an instance of a position encoder
        self.position_encoder = PositionEncoder(embedding_dim=d_model,
                                                dropout=dropout)

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor):
        x_emb = self.position_encoder(self.embedder(x))
        x_processed = self.encoder_stack(src=x_emb,
                                         mask=None,
                                         src_key_padding_mask=padding_mask)
        return self.generator(x_processed)