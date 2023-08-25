import numpy as np
from pandas import DataFrame

from src.model.tcr_metric import BetaCdrLevenshtein


def test_calc_cdist_matrix(mock_data_df: DataFrame):
    model = BetaCdrLevenshtein()
    anchor_tcrs = mock_data_df.iloc[0:2]
    comparison_tcrs = mock_data_df.iloc[0:3]

    result = model.calc_cdist_matrix(anchor_tcrs, comparison_tcrs)
    expected = np.array([[0, 6, 7], [6, 0, 13]])

    assert np.array_equal(result, expected)


def test_calc_pdist_vector(mock_data_df: DataFrame):
    model = BetaCdrLevenshtein()
    result = model.calc_pdist_vector(mock_data_df)
    expected = np.array([6, 7, 13])

    assert np.array_equal(result, expected)
