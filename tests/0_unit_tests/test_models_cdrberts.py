import pytest
from src.datahandling.datasets import TCRDataset
from src.datahandling.dataloaders import TCRDataLoader
from src.datahandling.tokenisers import CDRTokeniser
from src.models import *
import torch


model_classes = (CDRBERT,)
model_instances = [
    Model(name="foobar", num_encoder_layers=6, d_model=64, nhead=8, dim_feedforward=256)
    for Model in model_classes
]


@pytest.fixture
def cdrt_dataloader(mock_data_df):
    return TCRDataLoader(
        TCRDataset(
            mock_data_df, CDRTokeniser(p_drop_aa=0, p_drop_cdr=0, p_drop_chain=0)
        ),
        batch_size=2,
        shuffle=False,
    )


@pytest.mark.parametrize("model", model_instances)
class TestModel:
    def test_name(self, model):
        assert model.name == "foobar"

    def test_forward(self, model):
        batch = torch.zeros((3, 10, 4), dtype=torch.long)
        out, padding_mask = model(x=batch)

        assert out.size() == (3, 10, 64)
        assert padding_mask.size() == (3, 10)
        assert (padding_mask == 1).all()

    def test_embed(self, model, cdrt_dataloader):
        batch = next(iter(cdrt_dataloader))
        out = model.embed(x=batch)

        assert out.size() == (2, 64)
        torch.testing.assert_close(out.norm(dim=1), torch.ones(2))

    def test_mlm(self, model):
        batch = torch.zeros((3, 10, 4), dtype=torch.long)
        out = model.mlm(x=batch)

        assert out.size() == (3, 10, 20)
