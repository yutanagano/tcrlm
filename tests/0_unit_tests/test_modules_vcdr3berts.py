import pytest
from src.datahandling.datasets import TCRDataset
from src.datahandling.dataloaders import TCRDataLoader
from src.datahandling.tokenisers import BVCDR3Tokeniser
from src.modules import *
import torch


model_classes = (
    BVCDR3BERT,
    BVCDR3ClsBERT
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


@pytest.fixture
def bvcdr3t_dataloader(mock_data_beta_df):
    return TCRDataLoader(
        TCRDataset(mock_data_beta_df, BVCDR3Tokeniser()),
        batch_size=2,
        shuffle=False
    )


@pytest.mark.parametrize('model', model_instances)
class TestModel:
    def test_name(self, model):
        assert model.name == 'foobar'


    def test_forward(self, model):
        batch = torch.zeros((3,10,3), dtype=torch.long)
        out, padding_mask = model(x=batch)

        assert out.size() == (3,10,64)
        assert padding_mask.size() == (3,10)
        assert (padding_mask == 1).all()


    def test_embed(self, model, bvcdr3t_dataloader):
        batch = next(iter(bvcdr3t_dataloader))
        out = model.embed(x=batch)

        assert out.size() == (2,64)
        torch.testing.assert_close(out.norm(dim=1), torch.ones(2))


    def test_mlm(self, model):
        batch = torch.zeros((3,10,3), dtype=torch.long)
        out = model.mlm(x=batch)

        assert out.size() == (3,10,68)