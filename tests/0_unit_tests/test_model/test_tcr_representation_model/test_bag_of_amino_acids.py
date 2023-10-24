import math
from numpy import ndarray
import pytest
from src.model.tcr_representation_model import (
    AlphaCdr3BagOfAminoAcids,
    BetaCdr3BagOfAminoAcids,
    Cdr3BagOfAminoAcids,
    AlphaBagOfAminoAcids,
    BetaBagOfAminoAcids,
    BagOfAminoAcids,
)
from src.schema import AminoAcid


@pytest.mark.filterwarnings("ignore:Converting mask without torch.bool")
@pytest.mark.parametrize(
    "model",
    (
        AlphaCdr3BagOfAminoAcids(),
        BetaCdr3BagOfAminoAcids(),
        Cdr3BagOfAminoAcids(),
        AlphaBagOfAminoAcids(),
        BetaBagOfAminoAcids(),
        BagOfAminoAcids(),
    ),
)
class TestBagOfAminoAcids:
    def test_calc_cdist_matrix(self, model, mock_data_df):
        NUM_ANCHOR_TCRS = 2
        NUM_COMPARISON_TCRS = 3

        anchor_tcrs = mock_data_df.iloc[0:NUM_ANCHOR_TCRS]
        comparison_tcrs = mock_data_df.iloc[0:NUM_COMPARISON_TCRS]

        result = model.calc_cdist_matrix(anchor_tcrs, comparison_tcrs)

        assert isinstance(result, ndarray)
        assert result.shape == (NUM_ANCHOR_TCRS, NUM_COMPARISON_TCRS)

    def test_calc_pdist_vector(self, model, mock_data_df):
        result = model.calc_pdist_vector(mock_data_df)
        num_tcrs = len(mock_data_df)
        num_pairs = math.comb(num_tcrs, 2)

        assert isinstance(result, ndarray)
        assert result.shape == (num_pairs,)

    def test_calc_vector_representations(self, model, mock_data_df):
        D_MODEL = len(AminoAcid)
        result = model.calc_vector_representations(mock_data_df)
        num_tcrs = len(mock_data_df)

        assert isinstance(result, ndarray)
        assert result.shape == (num_tcrs, D_MODEL)
