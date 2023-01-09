import pytest
from src.modules import CDR3BERT_ap
import torch


@pytest.fixture
def cdr3bert_ap():
    model = CDR3BERT_ap(
        num_encoder_layers=6,
        d_model=64,
        nhead=8,
        dim_feedforward=256
    )

    return model


class TestCDR3BERT_a:
    def test_init_attributes(self, cdr3bert_ap):
        assert cdr3bert_ap.embed_layer == 5
        assert cdr3bert_ap._num_layers == 6
        assert cdr3bert_ap._d_model == 64
        assert cdr3bert_ap._nhead == 8
        assert cdr3bert_ap._dim_feedforward == 256


    def test_forward(self, cdr3bert_ap):
        batch = torch.zeros((3,10,3), dtype=torch.long)
        out, padding_mask = cdr3bert_ap(x=batch)

        assert out.size() == (3,10,64)
        assert padding_mask.size() == (3,10)
        assert (padding_mask == 1).all()


    def test_embed(self, cdr3bert_ap):
        batch = torch.tensor(
            [
                [[3,1,1],[4,1,2],[5,1,3],[3,2,1],[4,2,2],[5,2,3]],
                [[6,1,1],[7,1,2],[8,1,3],[6,2,1],[7,2,2],[8,2,3]],
                [[3,1,1],[4,1,2],[5,1,3],[6,2,1],[7,2,2],[8,2,3]]
            ],
            dtype=torch.long
        )
        out = cdr3bert_ap.embed(x=batch)

        assert out.size() == (3,64)
        torch.testing.assert_close(out.norm(dim=1), torch.ones(3))


    def test_mlm(self, cdr3bert_ap):
        batch = torch.zeros((3,10,3), dtype=torch.long)
        out = cdr3bert_ap.mlm(x=batch)

        assert out.size() == (3,10,20)


    def test_name(self, cdr3bert_ap):
        assert cdr3bert_ap.name == 'CDR3BERT_ap_6_64_8_256-embed_5'