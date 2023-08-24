import math
from numpy import ndarray
import pytest
import torch

from src.model.tcr_representation_model import Blastr
from src.data.tokeniser import BetaCdrTokeniser


def test_calc_cdist_matrix(blastr: Blastr, mock_data_df):
    NUM_ANCHOR_TCRS = 2
    NUM_COMPARISON_TCRS = 3

    anchor_tcrs = mock_data_df.iloc[0:NUM_ANCHOR_TCRS]
    comparison_tcrs = mock_data_df.iloc[0:NUM_COMPARISON_TCRS]

    result = blastr.calc_cdist_matrix(anchor_tcrs, comparison_tcrs)

    assert isinstance(result, ndarray)
    assert result.shape == (NUM_ANCHOR_TCRS, NUM_COMPARISON_TCRS)


def test_calc_pdist_vector(blastr: Blastr, mock_data_df):
    result = blastr.calc_pdist_vector(mock_data_df)
    num_tcrs = len(mock_data_df)
    num_pairs = math.comb(num_tcrs, 2)

    assert isinstance(result, ndarray)
    assert result.shape == (num_pairs,)


def test_calc_vector_representations(blastr: Blastr, toy_bert_d_model, mock_data_df):
    result = blastr.calc_vector_representations(mock_data_df)
    num_tcrs = len(mock_data_df)

    assert isinstance(result, ndarray)
    assert result.shape == (num_tcrs, toy_bert_d_model)


@pytest.fixture
def blastr(tokeniser, toy_bert):
    return Blastr(
        name="foobar", tokeniser=tokeniser, bert=toy_bert, device=torch.device("cpu")
    )


@pytest.fixture
def tokeniser():
    return BetaCdrTokeniser()
