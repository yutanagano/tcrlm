import pandas as pd
from pandas import DataFrame
import pytest

from src.model.tcr_metric import BetaCdr3Levenshtein


@pytest.fixture
def mock_bg_data(mock_data_df):
    return pd.concat([mock_data_df] * 4)

@pytest.fixture
def mock_pgens():
    return DataFrame(
        data={
            "alpha_pgen": [1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            "beta_pgen": [1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        }
    )

@pytest.fixture
def mock_labelled_data_dict(mock_data_df):
    return {"mock": mock_data_df}

@pytest.fixture
def beta_cdr3_levenshtein_model():
    return BetaCdr3Levenshtein()