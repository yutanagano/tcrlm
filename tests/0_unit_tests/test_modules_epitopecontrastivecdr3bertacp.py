import pytest
from src.modules import EpContCDR3BERT_acp


@pytest.fixture
def autocontrastive_cdr3bert_acp():
    model = EpContCDR3BERT_acp(
        contrastive_loss_type='Test',
        num_encoder_layers=6,
        d_model=64,
        nhead=8,
        dim_feedforward=256
    )

    return model


class TestAutoContrastive_CDR3BERT_cp:
    def test_name(self, autocontrastive_cdr3bert_acp):
        assert autocontrastive_cdr3bert_acp.name ==\
            'EpitopeContrastive_CDR3BERT_Test_acp_6_64_8_256'