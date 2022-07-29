'''
Python module with classes that represent the code base for the BERT-based
neural network models that will be able to learn and process TCR beta-chain
CDR3 sequences.
'''


import math
import torch
from torch import nn
from typing import Tuple


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
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-5
    ):
        super(Cdr3Bert, self).__init__()

        self._d_model = d_model
        self._nhead = nhead
        self._dim_feedforward = dim_feedforward

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
        
        # Create a stack of num_layers * encoder layer, our main network
        self.encoder_stack = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Create an embedder that can take in a LongTensor representing padded
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


    @property
    def d_model(self) -> int:
        return self._d_model
    

    @property
    def nhead(self) -> int:
        return self._nhead


    @property
    def dim_feedforward(self) -> int:
        return self._dim_feedforward


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


class TcrEmbedder(nn.Module):
    '''
    Neural network combination that takes two Cdr3Bert models, one pre-trained
    on alpha CDR3s, and another pre-trained on beta CDR3s.
    '''
    def __init__(self, alpha_bert: Cdr3Bert, beta_bert: Cdr3Bert):
        # Ensure that the alpha and beta bert models share the same model shape
        assert(alpha_bert.d_model == beta_bert.d_model)
        assert(alpha_bert.nhead == beta_bert.nhead)
        assert(alpha_bert.dim_feedforward == beta_bert.dim_feedforward)

        super(TcrEmbedder, self).__init__()
        self._alpha_bert = alpha_bert
        self._beta_bert = beta_bert
    

    @property
    def d_model(self) -> int:
        return self._alpha_bert.d_model
    

    @property
    def alpha_bert(self) -> Cdr3Bert:
        return self._alpha_bert
    

    @property
    def beta_bert(self) -> Cdr3Bert:
        return self._beta_bert
    

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        alpha_embedding = self._alpha_bert.embed(x_a)
        beta_embedding = self._beta_bert.embed(x_b)

        return torch.cat((alpha_embedding, beta_embedding), dim=1)


class Cdr3BertPretrainWrapper(nn.Module):
    '''
    Wrapper to put around a Cdr3Bert instance during pretraining to streamline
    the forward pass. In the pretraining process, the model will be trained on
    masked amino-acid modelling, where a random subset of the tokens in the
    input sequence will be masked, and it is the network's job to predict what
    those masked tokens were using the remaining tokens as context.
    '''
    def __init__(self, bert: Cdr3Bert):
        super(Cdr3BertPretrainWrapper, self).__init__()
        self._bert = bert

        # The generator is a linear layer whose job is to take CDR3BERT's final
        # layer output, then project that onto a probability distribution over
        # the 20 possible amino acids.
        self.generator = nn.Linear(
            in_features=bert.d_model,
            out_features=20
        )
    

    @property
    def bert(self) -> Cdr3Bert:
        return self._bert


    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.generator(self._bert(x)[0])


class Cdr3BertFineTuneWrapper(nn.Module):
    '''
    Wrapper to put around a Cdr3Bert instance during finetuning to streamline
    the forward pass. In the finetuning process, the model will be trained by
    embedding two CDR3 sequences into two fixed-size vectors, and then running
    a concatenation of the two embeddings (plus a third difference vector)
    through a single linear layer without bias to classify whether the two
    CDR3s respond to the same epitope.
    '''
    def __init__(self, tcr_embedder: TcrEmbedder):
        super(Cdr3BertFineTuneWrapper, self).__init__()

        self._embedder = tcr_embedder
        self.classifier = nn.Linear(6 * tcr_embedder.d_model, 2, bias=False)


    @property
    def d_model(self) -> int:
        return self._embedder.d_model

    
    @property
    def embedder(self) -> TcrEmbedder:
        return self._embedder

    
    def forward(
        self,
        x_1a: torch.Tensor, x_1b: torch.Tensor,
        x_2a: torch.Tensor, x_2b: torch.Tensor
    ) -> torch.Tensor:
        '''
        Feed the model two batches of paired alpha-beta CDR3 sequences
        (henceforth referred to as 'receptors'), and have it predict whether
        each pair of receptors (pair: two receptors found at the same indices in
        the two input batches) responds to the same epitope or not.
        Input: Two batches of tokenised receptors       (size: (N,S) x 4)*
        Output: Prediction of epitope match             (size: N,1)*

        * Dimensions are as follows:
        N - number of items in batch i.e. batch size
        S - number of tokens in sequence i.e. sequence length
        '''
        x_1_embedding = self._embedder(x_1a, x_1b)
        x_2_embedding = self._embedder(x_2a, x_2b)
        difference = x_1_embedding - x_2_embedding

        combined = torch.cat((x_1_embedding, x_2_embedding, difference), dim=1)
        
        return self.classifier(combined)


    def custom_trainmode(self):
        self.train()
        self._embedder.eval()