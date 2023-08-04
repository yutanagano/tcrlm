import pytest
from torch import Tensor

from src.model.data.tokenisers.beta_cdr_tokeniser import BetaCdrTokeniser
from src.model.data.tcr_dataset import TcrDataset
from src.model.data.tcr_dataloader import TcrDataLoader


@pytest.fixture
def tcr_dataloader(mock_tcr):
    mock_tcrs = [mock_tcr] * 10
    dataset = TcrDataset(mock_tcrs, BetaCdrTokeniser())
    dataloader = TcrDataLoader(dataset)

    return dataloader

def test_iter(tcr_dataloader):
    BATCH_SIZE = 1
    TOKENISED_TCR_LENGTH = 18
    TOKEN_NUM_DIMS = 4

    for batch in tcr_dataloader:
        assert type(batch) == Tensor
        assert batch.dim() == 3
        assert batch.size(0) == BATCH_SIZE
        assert batch.size(1) == TOKENISED_TCR_LENGTH
        assert batch.size(2) == TOKEN_NUM_DIMS