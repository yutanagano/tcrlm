'''
cdr3bert.py
purpose: Python module with classes that represent the code base for the BERT-
         based neural network models that will be able to learn and process TCR
         beta-chain CDR3 sequences.
author: Yuta Nagano
ver: 2.1.0
'''


import math
import torch
from torch import nn


# Helper functions
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
    # Resize padding mask to make it broadcastable with token embeddings
    padding_mask = padding_mask.unsqueeze(-1)

    # Compute averages of token embeddings per sequence, ignoring padding tokens
    token_embeddings_masked = token_embeddings * padding_mask
    token_embeddings_summed = token_embeddings_masked.sum(1)
    token_embeddings_averaged = token_embeddings_summed / padding_mask.sum(1)

    return token_embeddings_averaged


# Classes
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.position_encoding[:, :x.size(1)])


class Cdr3Bert(nn.Module):
    '''
    Neural network based on the BERT architecture modified to process TCR beta-
    chain CDR3 sequences.
    '''
    def __init__(
        self,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        layer_norm_eps: float = 1e-5
    ):
        
        super(Cdr3Bert, self).__init__()

        # Create an instance of the encoder layer that we want
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True
        )
        
        # Create a stack of num_layers * encoder layer, this is our main network
        self.encoder_stack = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Create a fully-connected layer that will function as the final layer,
        # which projects the fully processed token embeddings onto a probability
        # distribution over all possible amino acid residues.
        self.generator = nn.Linear(
            in_features=d_model,
            out_features=20
        )
        
        # Create an embedder that can take in a LongTensor representing a padded
        # batch of cdr3 sequences, and output a similar FloatTensor with an
        # extra dimension representing the embedding dimension.
        self.embedder = AaEmbedder(embedding_dim=d_model)

        # Create an instance of a position encoder
        self.position_encoder = PositionEncoder(
            embedding_dim=d_model,
            dropout=dropout
        )

        # Use xavier uniform initialisation for rank-2+ parameter tensors
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        '''
        Forward method of the network.
        Input: Batched and tokenised cdr3 sequences (size: N,S)*
        Output:
        1. Batched sequences of token embeddings    (size: N,S,E)*
        2. Padding mask for potential further use   (size: N,S)*

        * Dimensions are as follows:
        N - number of items in batch i.e. batch size
        S - number of tokens in sequence i.e. sequence length
        E - number of dimensions in embedding
        '''
        padding_mask = create_padding_mask(x)

        # Create an embedding of the input tensor (with positional info)
        x_emb = self.position_encoder(self.embedder(x))

        # Run the embedded input through the bert stack
        out = self.encoder_stack(
            src=x_emb,
            mask=None,
            src_key_padding_mask=padding_mask
        )

        return out, padding_mask


    def fill_in(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Feed the model a batch of cdr3 sequences with certain amino acid
        residues masked, and have the model generate a batch of sequences of
        token probability distributions for those masked tokens.
        Input: Batched and tokenised cdr3 sequences         (size: N,S)*
        Output: Batched sequences of token probabilities    (size: N,S,V)*

        * Dimensions are as follows:
        N - number of items in batch i.e. batch size
        S - number of tokens in sequence i.e. sequence length
        V - vocabulary size (in this case 20 for 20 amino acids)
        '''
        # Return the generator projections of the BERT output
        return self.generator(self.forward(x)[0])


    def embed(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Use the model to generate fix-sized vector embeddings of CDR3s by
        passing a vectorised representation of the CDR3 through the model, then
        pooling the output layer's per-token embeddings in some way (default:
        average pooling) to produce one vector embedding for the whole CDR3
        amino acid sequence.
        Input: Batched and tokenised cdr3 sequenecs             (size: N,S)*
        Output: Batched vector embeddings of the cdr3 sequences (size: N,E)*

        * Dimensions are as follows:
        N - number of items in batch i.e. batch size
        S - number of tokens in sequence i.e. sequence length
        E - number of dimensions in embedding
        '''
        # Run the input through the BERT stack
        token_embeddings, padding_mask = self.forward(x)

        # Compute the masked average pool of the token embeddings to produce
        # cdr3 embeddings, and return those
        return masked_average_pool(token_embeddings, padding_mask)


class Cdr3BertPretrainWrapper(nn.Module):
    '''
    Wrapper to put around a Cdr3Bert instance during pretraining to streamline
    the forward pass.
    '''
    def __init__(self, bert: Cdr3Bert):
        super(Cdr3BertPretrainWrapper, self).__init__()
        self.bert = bert


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bert.fill_in(x)