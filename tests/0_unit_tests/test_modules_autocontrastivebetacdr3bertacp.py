import pytest
from src.modules import AutoContBetaCDR3BERT_ap


@pytest.fixture
def autocontrastive_betacdr3bert_ap():
    model = AutoContBetaCDR3BERT_ap(
        contrastive_loss_type='Test',
        num_encoder_layers=6,
        d_model=64,
        nhead=8,
        dim_feedforward=256
    )

    return model


class TestModel:
    def test_name(self, autocontrastive_betacdr3bert_ap):
        assert autocontrastive_betacdr3bert_ap.name ==\
            'AutoContrastive_BetaCDR3BERT_Test_ap_6_64_8_256'