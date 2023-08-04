import pytest
from torch import Tensor

from src.model.data.tcr_dataset import TcrDataset
from src.model.data.tokenisers.tokeniser import Tokeniser
from src.tcr import Tcr


class MockTokeniser(Tokeniser):
    def tokenise(self, tcr: Tcr) -> Tensor:
        return tcr
    
    def tokenise_with_dropout(self, tcr: Tcr) -> Tensor:
        return tcr


@pytest.fixture
def tcr_dataset(mock_tcr):
    mock_tcr_list = [mock_tcr] * 10
    mock_tokeniser = MockTokeniser()
    return TcrDataset(mock_tcr_list, mock_tokeniser)

def test_len(tcr_dataset):
    assert len(tcr_dataset) == 10

def test_getitem(tcr_dataset, mock_tcr):
    assert tcr_dataset[0] == mock_tcr