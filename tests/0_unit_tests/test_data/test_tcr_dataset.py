import pytest

from src.data.tcr_dataset import TcrDataset
from src.data.tcr import Tcr, Tcrv, TravGene, TrbvGene
from src.data.pmhc import MhcGene, Pmhc


def test_len(tcr_dataset):
    assert len(tcr_dataset) == 3


def test_getitem(tcr_dataset):
    first_tcr_pmhc_pair = tcr_dataset[0]
    expected_tcr = Tcr(
        trav=Tcrv(TravGene["TRAV1-1"], 1),
        junction_a_sequence="CAVKASGSRLT",
        trbv=Tcrv(TrbvGene["TRBV2"], 1),
        junction_b_sequence="CASSDRAQPQHF",
    )
    expected_pmhc = Pmhc(
        epitope_sequence="CLAMP",
        mhc_a=MhcGene("HLA-A*01"),
        mhc_b=MhcGene("B2M")
    )

    assert first_tcr_pmhc_pair.tcr == expected_tcr
    assert first_tcr_pmhc_pair.pmhc == expected_pmhc


@pytest.fixture
def tcr_dataset(mock_data_df):
    return TcrDataset(mock_data_df)
