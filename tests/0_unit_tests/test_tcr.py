import pytest

from src.tcr import TravGene, TrbvGene, Tcrv, Tcr


@pytest.fixture
def example_trav():
    return Tcrv(gene=TravGene["TRAV1-1"], allele_num=1)


@pytest.fixture
def example_trbv():
    return Tcrv(gene=TrbvGene["TRBV3-1"], allele_num=1)


@pytest.fixture
def example_tcr(example_trav, example_trbv):
    example_junction_a = "CASQYF"
    example_junction_b = "CATQYF"

    return Tcr(
        trav=example_trav,
        junction_a_sequence=example_junction_a,
        trbv=example_trbv,
        junction_b_sequence=example_junction_b,
    )


class TestTcrv:
    def test_cdr1_sequence(self, example_trav: Tcrv):
        result = example_trav.cdr1_sequence
        expected = "TSGFYG"

        assert result == expected

    def test_cdr2_sequence(self, example_trbv: Tcrv):
        result = example_trbv.cdr2_sequence
        expected = "YNNKEL"

        assert result == expected

    def test_unknown_gene(self):
        mystery_tcrv = Tcrv(gene=None, allele_num=None)

        assert mystery_tcrv.cdr1_sequence == None
        assert mystery_tcrv.cdr2_sequence == None

    def test_assume_first_allele_if_not_known(self):
        mystery_allele = Tcrv(gene="TRAV1-1", allele_num=None)

        assert mystery_allele.gene == "TRAV1-1"
        assert mystery_allele.allele_num == 1

    def test_equality(self):
        anchor = Tcrv(gene=TravGene["TRAV23/DV6"], allele_num=1)
        comparison = Tcrv(gene=TravGene["TRAV23/DV6"], allele_num=1)

        assert anchor == comparison

    def test_repr(self, example_trav: Tcrv):
        result = repr(example_trav)
        expected = "TRAV1-1*01"

        assert result == expected

    def test_repr_when_unknown(self):
        mystery_tcrv = Tcrv(gene=None, allele_num=None)
        result = repr(mystery_tcrv)
        expected = "?"

        assert result == expected


class TestTcr:
    def test_cdr1a_sequence(self, example_tcr: Tcr):
        result = example_tcr.cdr1a_sequence
        expected = "TSGFYG"

        assert result == expected

    def test_cdr1b_sequence(self, example_tcr: Tcr):
        result = example_tcr.cdr1b_sequence
        expected = "LGHDT"

        assert result == expected

    def test_cdr2a_sequence(self, example_tcr: Tcr):
        result = example_tcr.cdr2a_sequence
        expected = "NALDGL"

        assert result == expected

    def test_cdr2b_sequence(self, example_tcr: Tcr):
        result = example_tcr.cdr2b_sequence
        expected = "YNNKEL"

        assert result == expected

    def test_junction_a_sequence(self, example_tcr: Tcr):
        result = example_tcr.junction_a_sequence
        expected = "CASQYF"

        assert result == expected

    def test_junction_b_sequence(self, example_tcr: Tcr):
        result = example_tcr.junction_b_sequence
        expected = "CATQYF"

        assert result == expected

    def test_equality(self):
        anchor = Tcr(
            trav=Tcrv(gene=TravGene["TRAV1-1"], allele_num=1),
            junction_a_sequence="CASQYF",
            trbv=Tcrv(gene=TrbvGene["TRBV3-1"], allele_num=1),
            junction_b_sequence="CATQYF",
        )
        comparison = Tcr(
            trav=Tcrv(gene=TravGene["TRAV1-1"], allele_num=1),
            junction_a_sequence="CASQYF",
            trbv=Tcrv(gene=TrbvGene["TRBV3-1"], allele_num=1),
            junction_b_sequence="CATQYF",
        )

        assert anchor == comparison

    def test_repr(self, example_tcr: Tcr):
        result = repr(example_tcr)
        expected = "Tra(TRAV1-1*01, CASQYF), Trb(TRBV3-1*01, CATQYF)"

        assert result == expected
