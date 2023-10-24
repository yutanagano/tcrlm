from src.model_analyser.tcr_edit_distance_records.coverage_summary import (
    CoverageSummary,
)


def test_repr():
    coverage_summary = CoverageSummary(range(10))

    assert repr(coverage_summary) == "min: 0, max: 9, mean: 4.5"
