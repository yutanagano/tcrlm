import pytest

from src import data


@pytest.mark.parametrize(
    argnames=("anchor", "comparison", "expected"),
    argvalues=(
        (
            data.make_pmhc_from_components("AAA", "HLA-A*01:01", "B2M"),
            data.make_pmhc_from_components("AAA", "HLA-A*01:01", "B2M"),
            True
        ),
        (
            data.make_pmhc_from_components("AAA", "HLA-A*01:01", "B2M"),
            data.make_pmhc_from_components("CCC", "HLA-DRA*01:01", "HLA-DRB1*01:01"),
            False
        ),
        (
            data.make_pmhc_from_components("AAA", "HLA-A*01:01", "B2M"),
            data.make_pmhc_from_components("AAA", None, None),
            True
        ),
        (
            data.make_pmhc_from_components("CCC", "HLA-DRA*01:01", "HLA-DRB1*01:01"),
            data.make_pmhc_from_components("CCC", "HLA-DRA", "HLA-DRB1"),
            True
        ),
        (
            data.make_pmhc_from_components(None, "HLA-DRA*01:01", "HLA-DRB1*01:01"),
            data.make_pmhc_from_components(None, "HLA-DRA*01:01", "HLA-DRB1*01:01"),
            False
        ),
        (
            data.make_pmhc_from_components("AAA", None, None),
            data.make_pmhc_from_components("AAA", None, None),
            True
        ),
    )
)
def test_equality(anchor, comparison, expected):
    result = anchor == comparison
    assert result == expected


def test_repr():
    pmhc = data.make_pmhc_from_components("AAA", "HLA-A*01:01", "B2M")
    assert repr(pmhc) == "AAA/HLA-A*01:01/B2M"