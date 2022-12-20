'''
CDR3BERT classes
'''


from src.modules.bert.bert import BERT_base
from src.modules.bert.embedding import AAEmbedding_c, AAEmbedding_cp
import torch
from torch import Tensor
from torch.nn.functional import normalize


class CDR3BERT_c(BERT_base):
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
        dropout: float = 0.1
    ) -> None:
        super().__init__(
            num_encoder_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )

        self.embedder = AAEmbedding_c(embedding_dim=d_model)
        self.generator = torch.nn.Linear(d_model, 20)


    @property
    def name(self) -> str:
        return f'CDR3BERT_c_{self._num_layers}_{self._d_model}_'\
            f'{self._nhead}_{self._dim_feedforward}-embed_{self.embed_layer}'


class CDR3BERT_cp(BERT_base):
    '''
    CDR3BERT model that get amino acid, chain, and residue position
    information.

    Compatible tokenisers: CDR3Tokeniser
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

        self.embedder = AAEmbedding_cp(embedding_dim=d_model)
        self.generator = torch.nn.Linear(d_model, 20)


    @property
    def name(self) -> str:
        return f'CDR3BERT_cp_{self._num_layers}_{self._d_model}_'\
            f'{self._nhead}_{self._dim_feedforward}-embed_{self.embed_layer}'


class _CLS_CDR3BERT_cp(CDR3BERT_cp):
    '''
    CDR3BERT_cp model that embeds using the <cls> token.

    Compatible tokenisers: CDR3Tokeniser
    '''
    def embed(self, x: Tensor) -> Tensor:
        '''
        Get the l2-normalised <cls> embeddings of the final layer.
        '''
        x_emb = self.forward(x)[0]
        x_emb = x_emb[:,0,:]
        return normalize(x_emb, p=2, dim=1)


class _Contrastive_CDR3BERT_cp(_CLS_CDR3BERT_cp):
    '''
    CLS_CDR3BERT_cp model with code to generate model name based on contrastive
    loss type.

    Compatible tokenisers: CDR3Tokeniser
    '''
    _model_base_name = None


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

        self._loss_type = contrastive_loss_type

    
    @property
    def name(self) -> str:
        return f'{self._model_base_name}_{self._loss_type}_CDR3BERT_cp_'\
            f'{self._num_layers}_{self._d_model}_'\
            f'{self._nhead}_{self._dim_feedforward}'


class AutoContrastive_CDR3BERT_cp(_Contrastive_CDR3BERT_cp):
    '''
    Contrastive_CDR3BERT_cp model with name 'AutoContrastive...'

    Compatible tokenisers: CDR3Tokeniser
    '''
    _model_base_name = 'AutoContrastive'


class EpitopeContrastive_CDR3BERT_cp(_Contrastive_CDR3BERT_cp):
    '''
    Contrastive_CDR3BERT_cp model with name 'EpitopeContrastive...'

    Compatible tokenisers: CDR3TOkeniser
    '''
    _model_base_name = 'EpitopeContrastive'