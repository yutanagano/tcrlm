'''
CDR3BERT classes

Compatible tokenisers: ABCDR3Tokeniser, BCDR3Tokeniser
'''


from src.modules.bert.bert import _BERTBase, _BERTClsEmbedBase
from src.modules.bert.embedding import (
    AAEmbedding_a,
    AAEmbedding_ap,
    AAEmbedding_ac,
    AAEmbedding_apc
)
import torch


class _CDR3BERTBase(_BERTBase):
    '''
    CDR3BERT base class.
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


class CDR3BERT_a(_CDR3BERTBase):
    '''
    CDR3BERT model that only gets amino acid information.
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

        self.embedder = AAEmbedding_a(embedding_dim=d_model)


class CDR3BERT_ap(_CDR3BERTBase):
    '''
    CDR3BERT model that gets amino acid and positional information.
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

        self.embedder = AAEmbedding_ap(embedding_dim=d_model)


class CDR3BERT_ac(_CDR3BERTBase):
    '''
    CDR3BERT model that gets amino acid and chain information.
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

        self.embedder = AAEmbedding_ac(embedding_dim=d_model)


class CDR3BERT_apc(_CDR3BERTBase):
    '''
    CDR3BERT model that gets amino acid, chain, and residue position
    information.
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

        self.embedder = AAEmbedding_apc(embedding_dim=d_model)


class CDR3ClsBERT_ap(_BERTClsEmbedBase, CDR3BERT_ap):
    '''
    CDR3BERT_ap model which uses the <cls> token to embed.
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


class CDR3ClsBERT_apc(_BERTClsEmbedBase, CDR3BERT_apc):
    '''
    CDR3BERT_acp model which uses the <cls> token to embed.
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