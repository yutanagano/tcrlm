import pytest
from src.modules import CDR3BERT_a
import torch


@pytest.fixture
def model():
    model = CDR3BERT_a(
        num_encoder_layers=6,
        d_model=64,
        nhead=8,
        dim_feedforward=256
    )

    return model


class TestModel:
    def test_init_attributes(self, model):
        assert model.embed_layer == 5
        assert model._num_layers == 6
        assert model._d_model == 64
        assert model._nhead == 8
        assert model._dim_feedforward == 256


    def test_forward(self, model):
        batch = torch.zeros((3,10,3), dtype=torch.long)
        out, padding_mask = model(x=batch)

        assert out.size() == (3,10,64)
        assert padding_mask.size() == (3,10)
        assert (padding_mask == 1).all()


    def test_embed(self, model):
        batch = torch.tensor(
            [
                [[3,1,1],[4,1,2],[5,1,3],[3,2,1],[4,2,2],[5,2,3]],
                [[6,1,1],[7,1,2],[8,1,3],[6,2,1],[7,2,2],[8,2,3]],
                [[3,1,1],[4,1,2],[5,1,3],[6,2,1],[7,2,2],[8,2,3]]
            ],
            dtype=torch.long
        )
        out = model.embed(x=batch)

        assert out.size() == (3,64)
        torch.testing.assert_close(out.norm(dim=1), torch.ones(3))


    def test_mlm(self, model):
        batch = torch.zeros((3,10,3), dtype=torch.long)
        out = model.mlm(x=batch)

        assert out.size() == (3,10,20)


    def test_name(self, model):
        assert model.name == 'CDR3BERT_a_6_64_8_256-embed_5'