import pytest
from torch import Tensor

from src.data.tcr_dataset import TcrDataset
from src.data.tokeniser.tokeniser import Tokeniser
from src.tcr import Tcr, Tcrv, TravGene, TrbvGene
from src.data.tokeniser.token_indices import AminoAcidTokenIndex


def test_len(tcr_dataset):
    assert len(tcr_dataset) == 3

def test_getitem(tcr_dataset):
    first_tcr = tcr_dataset[0]
    expected_tcr = Tcr(
        trav=Tcrv(TravGene["TRAV1-1"], 1),
        junction_a_sequence="CAVKASGSRLT",
        trbv=Tcrv(TrbvGene["TRBV2"], 1),
        junction_b_sequence="CASSDRAQPQHF"
    )

    assert first_tcr == expected_tcr


class MockTokeniser(Tokeniser):
    token_vocabulary_index = AminoAcidTokenIndex

    def tokenise(self, tcr: Tcr) -> Tensor:
        return tcr


@pytest.fixture
def tcr_dataset(mock_data_df):
    return TcrDataset(mock_data_df, MockTokeniser())