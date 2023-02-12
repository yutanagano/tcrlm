'''
CDRBERT classes

Compatible tokenisers: BCDRTokeniser
'''


from src.modules.bert.bert import _BERTBase, _BERTClsEmbedBase
from src.modules.bert.embedding.cdr import BCDREmbedding
import torch


class _CDRBERTBase(_BERTBase):
    '''
    CDRBERT base class.
    '''


    def __init__(
        self,
        name: str,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            name,
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )

        self.generator = torch.nn.Linear(d_model, 20)


class BCDRBERT(_CDRBERTBase):
    '''
    CDRBERT model for beta-chain only data.
    '''


    def __init__(
        self,
        name: str,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            name,
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )

        self.embedder = BCDREmbedding(embedding_dim=d_model)


class BCDRClsBERT(_BERTClsEmbedBase, BCDRBERT):
    '''
    BCDRBERT model which uses the <cls> token to embed.
    '''


    def __init__(
        self,
        name: str,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            name,
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )