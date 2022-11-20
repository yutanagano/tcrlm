'''
BERT templates.
'''
from src.modules.embedder import MLMEmbedder
from src.utils import masked_average_pool
import torch
from torch import Tensor
from torch.nn.functional import normalize
from typing import Tuple


class BERT_base(MLMEmbedder):
    '''
    BERT base template.
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

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder_stack = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )


    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        padding_mask = (x[:,:,0] == 0)

        # Create an embedding of the input tensor, and run through BERT stack
        x_emb = self.embedder(x)
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
        emb_mask = padding_mask.clone()
        emb_mask[:,0] = 0
        x_emb = masked_average_pool(x_emb, padding_mask)

        # l2 norm and return
        return normalize(x_emb, p=2, dim=1)


    def mlm(self, x: Tensor) -> Tensor:
        return self.generator(self.forward(x)[0])