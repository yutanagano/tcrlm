import pytest
from src.modules import EpitopeContrastive_CDR3BERT_cp


@pytest.fixture
def autocontrastive_cdr3bert_cp():
    model = EpitopeContrastive_CDR3BERT_cp(
        contrastive_loss_type='Test',
        num_encoder_layers=6,
        d_model=64,
        nhead=8,
        dim_feedforward=256
    )

    return model


class TestAutoContrastive_CDR3BERT_cp:
    def test_name(self, autocontrastive_cdr3bert_cp):
        assert autocontrastive_cdr3bert_cp.name ==\
            'EpitopeContrastive_Test_CDR3BERT_cp_6_64_8_256'