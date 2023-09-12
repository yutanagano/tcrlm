from typing import Iterable


class CoverageSummary:
    def __init__(self, num_estimates_made_per_edit_group: Iterable) -> None:
        self.lowest_coverage_level = min(num_estimates_made_per_edit_group)
        self.highest_coverage_level = max(num_estimates_made_per_edit_group)
        self.average_coverage_level = sum(num_estimates_made_per_edit_group) / len(
            num_estimates_made_per_edit_group
        )

    def __repr__(self) -> str:
        return f"min: {self.lowest_coverage_level}, max: {self.highest_coverage_level}, mean: {self.average_coverage_level}"
