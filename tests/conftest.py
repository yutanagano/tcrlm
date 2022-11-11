import pandas as pd
from pathlib import Path
import pytest
from src.datahandling.datasets import TCRDataset
from src.datahandling.tokenisers import CDR3Tokeniser


@pytest.fixture
def mock_data_path():
    return Path('tests/resources/mock_data.csv')


@pytest.fixture
def mock_data_df(mock_data_path):
    df = pd.read_csv(mock_data_path, dtype='string')
    return df


@pytest.fixture
def cdr3t_dataset(mock_data_path):
    dataset = TCRDataset(
        data=mock_data_path,
        tokeniser=CDR3Tokeniser()
    )

    return dataset