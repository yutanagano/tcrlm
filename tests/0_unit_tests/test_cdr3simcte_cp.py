import pytest
from src.modules import SimCTE_CDR3BERT_cp
import torch


@pytest.fixture
def simcte_cdr3bert_cp():
    model = SimCTE_CDR3BERT_cp(
        num_encoder_layers=6,
        d_model=64,
        nhead=8,
        dim_feedforward=256
    )

    return model


class TestSimCTE_CDR3BERT_cp:
    def test_init_attributes(self, simcte_cdr3bert_cp):
        assert simcte_cdr3bert_cp._num_layers == 6
        assert simcte_cdr3bert_cp._d_model == 64
        assert simcte_cdr3bert_cp._nhead == 8
        assert simcte_cdr3bert_cp._dim_feedforward == 256


    def test_forward(self, simcte_cdr3bert_cp):
        batch = torch.zeros((3,10,3), dtype=torch.long)
        out, padding_mask = simcte_cdr3bert_cp(x=batch)

        assert out.size() == (3,10,64)
        assert padding_mask.size() == (3,10)
        assert (padding_mask == 1).all()


    def test_embed(self, simcte_cdr3bert_cp):
        batch = torch.tensor(
            [
                [[2,0,0],[3,1,1],[4,1,2],[5,1,3],[3,2,1],[4,2,2],[5,2,3]],
                [[2,0,0],[6,1,1],[7,1,2],[8,1,3],[6,2,1],[7,2,2],[8,2,3]],
                [[2,0,0],[3,1,1],[4,1,2],[5,1,3],[6,2,1],[7,2,2],[8,2,3]]
            ],
            dtype=torch.long
        )
        out = simcte_cdr3bert_cp.embed(x=batch)

        assert out.size() == (3,64)
        torch.testing.assert_close(out.norm(dim=1), torch.ones(3))


    def test_mlm(self, simcte_cdr3bert_cp):
        batch = torch.zeros((3,10,3), dtype=torch.long)
        out = simcte_cdr3bert_cp.mlm(x=batch)

        assert out.size() == (3,10,20)


    def test_name(self, simcte_cdr3bert_cp):
        assert simcte_cdr3bert_cp.name == 'SimCTE_CDR3BERT_cp_6_64_8_256'