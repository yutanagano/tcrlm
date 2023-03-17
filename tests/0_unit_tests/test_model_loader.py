import pandas as pd
from pathlib import Path
import pytest
from src.model_loader import ModelLoader


@pytest.fixture
def model():
    return ModelLoader(Path("tests") / "resources" / "BCDR3BERT")


class TestModelLoader:
    def test_name(self, model):
        assert model.name == "BCDR3BERT"

    def test_embed(self, model, mock_data_beta_df):
        embeddings = model.embed(mock_data_beta_df)

        assert embeddings.shape == (2, 128)

    def test_pdist(self, model, mock_data_beta_df):
        pdist = model.pdist(mock_data_beta_df)

        assert pdist.shape == (1,)

    def test_cdist(self, model, mock_data_beta_df):
        cdist = model.cdist(mock_data_beta_df, mock_data_beta_df)

        assert cdist.shape == (2, 2)
