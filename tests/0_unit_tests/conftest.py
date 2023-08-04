import pandas as pd
from pathlib import Path
import pytest

from src.tcr import TravGene, TrbvGene, Tcrv, Tcr
from src.datahandling import datasets
from src.datahandling.dataloaders import TCRDataLoader
from src.datahandling.tokenisers import CDR3Tokeniser


@pytest.fixture
def mock_tcr():
    trav = Tcrv(TravGene["TRAV1-1"], 1)
    trbv = Tcrv(TrbvGene["TRBV2"], 1)
    tcr = Tcr(trav, "CATQYF", trbv, "CASQYF")

    return tcr

@pytest.fixture(scope="session")
def mock_data_path():
    return Path("tests") / "resources" / "mock_data.csv"


@pytest.fixture(scope="session")
def mock_data_df(mock_data_path):
    df = pd.read_csv(
        mock_data_path,
        dtype={
            "TRAV": "string",
            "CDR3A": "string",
            "TRAJ": "string",
            "TRBV": "string",
            "CDR3B": "string",
            "TRBJ": "string",
            "Epitope": "string",
            "MHCA": "string",
            "MHCB": "string",
            "duplicate_count": "UInt32",
        },
    )
    return df


@pytest.fixture(scope="session")
def mock_data_beta_df():
    df = pd.read_csv(
        Path("tests") / "resources" / "mock_data_beta.csv",
        dtype={
            "TRAV": "string",
            "CDR3A": "string",
            "TRAJ": "string",
            "TRBV": "string",
            "CDR3B": "string",
            "TRBJ": "string",
            "Epitope": "string",
            "MHCA": "string",
            "MHCB": "string",
            "duplicate_count": "UInt32",
        },
    )
    return df


@pytest.fixture(scope="session")
def mock_data_large_df():
    df = pd.read_csv(
        Path("tests") / "resources" / "mock_data_large.csv",
        dtype={
            "TRAV": "string",
            "CDR3A": "string",
            "TRAJ": "string",
            "TRBV": "string",
            "CDR3B": "string",
            "TRBJ": "string",
            "Epitope": "string",
            "MHCA": "string",
            "MHCB": "string",
            "duplicate_count": "UInt32",
        },
    )
    return df


@pytest.fixture
def abcdr3t_dataset(mock_data_df):
    dataset = datasets.TCRDataset(data=mock_data_df, tokeniser=CDR3Tokeniser())

    return dataset


@pytest.fixture
def abcdr3t_contrastive_dataset(mock_data_df):
    dataset = datasets.AutoContrastiveDataset(
        data=mock_data_df,
        tokeniser=CDR3Tokeniser(),
        censoring_lhs=False,
        censoring_rhs=True,
    )

    return dataset


@pytest.fixture
def abcdr3t_epitope_contrastive_dataset(mock_data_df):
    dataset = datasets.EpitopeContrastiveDataset_dep(
        data=mock_data_df,
        tokeniser=CDR3Tokeniser(),
        censoring_lhs=False,
        censoring_rhs=False,
    )

    return dataset


@pytest.fixture
def abcdr3t_dataloader(abcdr3t_dataset):
    dl = TCRDataLoader(dataset=abcdr3t_dataset, batch_size=3, shuffle=False)
    return dl
