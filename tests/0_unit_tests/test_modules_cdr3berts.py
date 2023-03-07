import pytest
from src.modules import *
import torch


model_classes = (
    CDR3BERT_a,
    CDR3BERT_ac,
    CDR3BERT_ap,
    CDR3BERT_ar,
    CDR3BERT_ab,
    CDR3BERT_apc,
    CDR3ClsBERT_ap,
    CDR3ClsBERT_ab,
    CDR3ClsBERT_apc
)
model_instances = [
    Model(
        name='foobar',
        num_encoder_layers=6,
        d_model=64,
        nhead=8,
        dim_feedforward=256
    ) for Model in model_classes
]


@pytest.mark.parametrize('model', model_instances)
class TestModel:
    def test_name(self, model):
        assert model.name == 'foobar'


    def test_forward(self, model):
        batch = torch.zeros((3,10,4), dtype=torch.long)
        out, padding_mask = model(x=batch)

        assert out.size() == (3,10,64)
        assert padding_mask.size() == (3,10)
        assert (padding_mask == 1).all()


    def test_embed(self, model, abcdr3t_dataloader):
        batch = next(iter(abcdr3t_dataloader))
        out = model.embed(x=batch)

        assert out.size() == (3,64)
        torch.testing.assert_close(out.norm(dim=1), torch.ones(3))


    def test_mlm(self, model):
        batch = torch.zeros((3,10,4), dtype=torch.long)
        out = model.mlm(x=batch)

        assert out.size() == (3,10,20)