import pandas as pd
from pathlib import Path
import pytest

from src.nn.data import schema


@pytest.fixture
def mock_tcr():
    return schema.make_tcr_from_components("TRAV1-1*01", "CATQYF", "TRBV2*01", "CASQYF")


@pytest.fixture
def mock_data_path():
    return Path("tests") / "resources" / "mock_data.csv"


@pytest.fixture
def mock_data_df(mock_data_path):
    return pd.read_csv(mock_data_path)
