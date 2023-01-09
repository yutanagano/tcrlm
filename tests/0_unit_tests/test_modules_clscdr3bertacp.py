import pytest
from src.modules.bert.cdr3bert import _CLS_CDR3BERT_acp
import torch


@pytest.fixture
def cls_cdr3bert_acp():
    model = _CLS_CDR3BERT_acp(
        num_encoder_layers=6,
        d_model=64,
        nhead=8,
        dim_feedforward=256
    )

    return model


class TestCLS_CDR3BERT_cp:
    def test_init_attributes(self, cls_cdr3bert_acp):
        assert cls_cdr3bert_acp._num_layers == 6
        assert cls_cdr3bert_acp._d_model == 64
        assert cls_cdr3bert_acp._nhead == 8
        assert cls_cdr3bert_acp._dim_feedforward == 256


    def test_forward(self, cls_cdr3bert_acp):
        batch = torch.zeros((3,10,3), dtype=torch.long)
        out, padding_mask = cls_cdr3bert_acp(x=batch)

        assert out.size() == (3,10,64)
        assert padding_mask.size() == (3,10)
        assert (padding_mask == 1).all()


    def test_embed(self, cls_cdr3bert_acp):
        batch = torch.tensor(
            [
                [[2,0,0],[3,1,1],[4,1,2],[5,1,3],[3,2,1],[4,2,2],[5,2,3]],
                [[2,0,0],[6,1,1],[7,1,2],[8,1,3],[6,2,1],[7,2,2],[8,2,3]],
                [[2,0,0],[3,1,1],[4,1,2],[5,1,3],[6,2,1],[7,2,2],[8,2,3]]
            ],
            dtype=torch.long
        )
        out = cls_cdr3bert_acp.embed(x=batch)

        assert out.size() == (3,64)
        torch.testing.assert_close(out.norm(dim=1), torch.ones(3))


    def test_mlm(self, cls_cdr3bert_acp):
        batch = torch.zeros((3,10,3), dtype=torch.long)
        out = cls_cdr3bert_acp.mlm(x=batch)

        assert out.size() == (3,10,20)