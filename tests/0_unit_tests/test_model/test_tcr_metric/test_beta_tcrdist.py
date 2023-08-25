import numpy as np
from pandas import DataFrame

from src.model.tcr_metric import BetaTcrdist


def test_calc_cdist_matrix(mock_data_df: DataFrame):
    model = BetaTcrdist()
    anchor_tcrs = mock_data_df.iloc[0:2]
    comparison_tcrs = mock_data_df.iloc[0:3]

    result = model.calc_cdist_matrix(anchor_tcrs, comparison_tcrs)
    expected = np.array([[0, 69, 46], [69, 0, 115]])

    assert np.array_equal(result, expected)


def test_calc_pdist_vector(mock_data_df: DataFrame):
    model = BetaTcrdist()
    result = model.calc_pdist_vector(mock_data_df)
    expected = np.array([69, 46, 115])

    assert np.array_equal(result, expected)
