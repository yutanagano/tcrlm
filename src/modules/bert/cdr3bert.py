'''
CDR3BERT classes

Compatible tokenisers: CDR3Tokeniser
'''


from src.modules.bert.bert import BERTBase, BERTClsEmbedBase
from src.modules.bert.embedding import (
    AAEmbedding_a,
    AAEmbedding_ap,
    AAEmbedding_ac,
    AAEmbedding_acp
)
import torch
from typing import Optional


class _CDR3BERTBase(BERTBase):
    '''
    CDR3BERT base class.
    '''


    _name_base = 'CDR3BERT'


    def __init__(
        self,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )

        self.generator = torch.nn.Linear(d_model, 20)


class _BetaCDR3BERTBase(_CDR3BERTBase):
    '''
    Base class variant for models specialising in beta chain.
    '''


    def __init__(
        self,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )

        self._name_base = 'Beta' + self._name_base


class CDR3BERT_a(_CDR3BERTBase):
    '''
    CDR3BERT model that only gets amino acid information.
    '''


    _name_suffix = 'a'


    def __init__(
        self,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )

        self.embedder = AAEmbedding_a(embedding_dim=d_model)


class BetaCDR3BERT_a(_BetaCDR3BERTBase, CDR3BERT_a):
    '''Beta variant of CDR3BERT_a'''


    def __init__(
        self,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )


class CDR3BERT_ap(_CDR3BERTBase):
    '''
    CDR3BERT model that gets amino acid and positional information.
    '''


    _name_suffix = 'ap'


    def __init__(
        self,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )

        self.embedder = AAEmbedding_ap(embedding_dim=d_model)


class BetaCDR3BERT_ap(_BetaCDR3BERTBase, CDR3BERT_ap):
    '''Beta variant of CDR3BERT_ap'''


    def __init__(
        self,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        contrastive_loss_type: Optional[str] = None # dummy parameter for compatibility with autocontrastive pipeline
    ) -> None:
        super().__init__(
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )


class CDR3BERT_ac(_CDR3BERTBase):
    '''
    CDR3BERT model that gets amino acid and chain information.
    '''


    _name_suffix = 'ac'


    def __init__(
        self,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )

        self.embedder = AAEmbedding_ac(embedding_dim=d_model)


class CDR3BERT_acp(_CDR3BERTBase):
    '''
    CDR3BERT model that gets amino acid, chain, and residue position
    information.
    '''


    _name_suffix = 'acp'


    def __init__(
        self,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )

        self.embedder = AAEmbedding_acp(embedding_dim=d_model)


class _ConCDR3BERTBase_acp(BERTClsEmbedBase, CDR3BERT_acp):
    '''
    CDR3BERT_acp model which uses the <cls> token to embed.
    '''


    def __init__(
        self,
        contrastive_loss_type: str,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )

        self._name_base = self._name_base + '_' + contrastive_loss_type


class _ConBetaCDR3BERTBase_ap(BERTClsEmbedBase, BetaCDR3BERT_ap):
    '''
    BetaCDR3BERT_ap model which uses the <cls> token to embed.
    '''


    def __init__(
        self,
        contrastive_loss_type: str,
        num_encoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )

        self._name_base = self._name_base.lstrip('Beta')
        self._name_base = self._name_base + '_' + contrastive_loss_type


class AutoContCDR3BERT_acp(_ConCDR3BERTBase_acp):
    '''
    CDR3BERT_acp model embedding using the <cls> token and with base name
    'AutoContrastive...'
    '''

    
    _name_base = 'AutoContrastive_CDR3BERT'


class AutoContBetaCDR3BERT_ap(_ConBetaCDR3BERTBase_ap):
    '''
    BetaCDR3BERT_ap model embedding using the <cls> token and with base name
    'AutoContrastive...'
    '''

    
    _name_base = 'AutoContrastive_BetaCDR3BERT'


class EpContCDR3BERT_acp(_ConCDR3BERTBase_acp):
    '''
    CDR3BERT_acp model embedding using the <cls> token and with base name
    'EpitopeContrastive...'
    '''


    _name_base = 'EpitopeContrastive_CDR3BERT'


class EpContBetaCDR3BERT_ap(_ConBetaCDR3BERTBase_ap):
    '''
    BetaCDR3BERT_ap model embedding using the <cls> token and with base name
    'EpitopeContrastive...'
    '''

    
    _name_base = 'EpitopeContrastive_BetaCDR3BERT'