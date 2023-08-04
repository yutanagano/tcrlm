from src.tcr import TravGene, TrbvGene, Tcrv, Tcr


class TestTcrv:
    def test_get_cdr1_sequence(self):
        trav_1_1_01 = Tcrv(TravGene["TRAV1-1"], 1)
        cdr1_sequence = trav_1_1_01.get_cdr1_sequence()
        expected = "TSGFYG"

        assert cdr1_sequence == expected

    def test_get_cdr2_sequence(self):
        trbv_3_1_01 = Tcrv(TrbvGene["TRBV3-1"], 1)
        cdr2_sequence = trbv_3_1_01.get_cdr2_sequence()
        expected = "YNNKEL"

        assert cdr2_sequence == expected