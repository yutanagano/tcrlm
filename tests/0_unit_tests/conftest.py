import pandas as pd
from pathlib import Path
import pytest

from src.tcr import TravGene, TrbvGene, Tcrv, Tcr


@pytest.fixture
def mock_tcr():
    trav = Tcrv(TravGene["TRAV1-1"], 1)
    trbv = Tcrv(TrbvGene["TRBV2"], 1)
    tcr = Tcr(trav, "CATQYF", trbv, "CASQYF")

    return tcr

@pytest.fixture
def mock_data_path():
    return Path("tests") / "resources" / "mock_data.csv"

@pytest.fixture
def mock_data_df(mock_data_path):
    return pd.read_csv(mock_data_path)