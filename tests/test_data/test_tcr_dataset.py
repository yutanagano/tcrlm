import pytest
from src.data.dataset import TcrDataset, EpitopeBackgroundTcrDataset
from libtcrlm import schema


class TestTcrDataset:
    @pytest.fixture
    def tcr_dataset(self, mock_data_df):
        return TcrDataset(mock_data_df)

    def test_len(self, tcr_dataset):
        assert len(tcr_dataset) == 3

    def test_getitem(self, tcr_dataset):
        first_tcr_pmhc_pair = tcr_dataset[0]
        expected_tcr = schema.make_tcr_from_components(
            "TRAV1-1*01", "CAVKASGSRLT", "TRBV2*01", "CASSDRAQPQHF"
        )
        expected_pmhc = schema.make_pmhc_from_components("CLAMP", "HLA-A*01", "B2M")

        assert first_tcr_pmhc_pair.tcr == expected_tcr
        assert first_tcr_pmhc_pair.pmhc == expected_pmhc


class TestEpitopeBackgroundDataset:
    @pytest.fixture
    def epitope_background_dataset(self, mock_data_df):
        mock_data_df.loc[1,"Epitope"] = None
        return EpitopeBackgroundTcrDataset(mock_data_df, 1)

    def test_len(self, epitope_background_dataset):
        assert len(epitope_background_dataset) == 4

    def test_getitem(self, epitope_background_dataset):
        first_tcr_pmhc_pair = epitope_background_dataset[0]
        expected_tcr = schema.make_tcr_from_components(
            "TRAV1-1*01", "CAVKASGSRLT", "TRBV2*01", "CASSDRAQPQHF"
        )
        expected_pmhc = schema.make_pmhc_from_components("CLAMP", "HLA-A*01", "B2M")

        assert first_tcr_pmhc_pair.tcr == expected_tcr
        assert first_tcr_pmhc_pair.pmhc == expected_pmhc

        second_tcr_pmhc_pair = epitope_background_dataset[1]
        expected_tcr = schema.make_tcr_from_components(
            "TRAV1-1*01", "CLANGSRLT", "TRBV2*01", "CASSANDRAF"
        )

        assert second_tcr_pmhc_pair.tcr == expected_tcr
        assert second_tcr_pmhc_pair.pmhc.epitope_sequence is None
        assert second_tcr_pmhc_pair.pmhc.mhc_a.symbol is None
        assert second_tcr_pmhc_pair.pmhc.mhc_b.symbol is None

        third_tcr_pmhc_pair = epitope_background_dataset[2]
        expected_tcr = schema.make_tcr_from_components(
            "TRAV5*01", "CAVKASGSRLT", "TRBV6-9*01", "CASSDRAQPQHF"
        )
        expected_pmhc = schema.make_pmhc_from_components("STEAK", "HLA-A*02", "B2M")

        assert third_tcr_pmhc_pair.tcr == expected_tcr
        assert third_tcr_pmhc_pair.pmhc == expected_pmhc

        expected_tcr = schema.make_tcr_from_components(
            "TRAV1-1*01", "CLANGSRLT", "TRBV2*01", "CASSANDRAF"
        )

        fourth_tcr_pmhc_pair = epitope_background_dataset[3]
        expected_tcr = schema.make_tcr_from_components(
            "TRAV1-1*01", "CLANGSRLT", "TRBV2*01", "CASSANDRAF"
        )

        assert fourth_tcr_pmhc_pair.tcr == expected_tcr
        assert fourth_tcr_pmhc_pair.pmhc.epitope_sequence is None
        assert fourth_tcr_pmhc_pair.pmhc.mhc_a.symbol is None
        assert fourth_tcr_pmhc_pair.pmhc.mhc_b.symbol is None