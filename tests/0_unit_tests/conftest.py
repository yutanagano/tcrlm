import pandas as pd
from pathlib import Path
import pytest
from src.datahandling import datasets
from src.datahandling.tokenisers import CDR3Tokeniser


@pytest.fixture
def mock_data_path():
    return Path('tests/resources/mock_data.csv')


@pytest.fixture
def mock_data_df(mock_data_path):
    df = pd.read_csv(
        mock_data_path,
        dtype={
            'TRAV': 'string',
            'CDR3A': 'string',
            'TRAJ': 'string',
            'TRBV': 'string',
            'CDR3B': 'string',
            'TRBJ': 'string',
            'Epitope': 'string',
            'MHCA': 'string',
            'MHCB': 'string',
            'duplicate_count': 'UInt32'
        }
    )
    return df


@pytest.fixture
def cdr3t_dataset(mock_data_path):
    dataset = datasets.TCRDataset(
        data=mock_data_path,
        tokeniser=CDR3Tokeniser()
    )

    return dataset


@pytest.fixture
def cdr3t_unsupsimcl_dataset(mock_data_path):
    dataset = datasets.UnsupervisedSimCLDataset(
        data=mock_data_path,
        tokeniser=CDR3Tokeniser()
    )

    return dataset


@pytest.fixture
def cdr3t_supsimcl_dataset(mock_data_path):
    dataset = datasets.SupervisedSimCLDataset(
        data=mock_data_path,
        tokeniser=CDR3Tokeniser()
    )

    return dataset