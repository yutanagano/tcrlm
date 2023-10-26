import itertools
import math
import numpy as np
from pandas import DataFrame
from src.clustering import ClusteringMachine
from typing import Set


class BruteForceClusteringMachine(ClusteringMachine):
    @property
    def name(self) -> str:
        return self._tcr_metric.name

    def cluster(
        self, tcrs: DataFrame, distance_threshold: float, batch_size: int = 1000
    ) -> Set:
        tcrs = tcrs.reset_index(drop=True)

        range_over_batch_indices = list(range(0, len(tcrs), batch_size))
        pairs_within_threshold = []

        for anchor_start_index, comparison_start_index in itertools.combinations_with_replacement(range_over_batch_indices, r=2):
            anchor_tcrs = tcrs.iloc[
                anchor_start_index : anchor_start_index + batch_size
            ]
            comparison_tcrs = tcrs.iloc[
                comparison_start_index : comparison_start_index + batch_size
            ]
            cdist_matrix = self._tcr_metric.calc_cdist_matrix(
                anchor_tcrs, comparison_tcrs
            )

            (anchor_indices, comparison_indices) = np.nonzero(
                cdist_matrix <= distance_threshold
            )
            anchor_indices += anchor_start_index
            comparison_indices += comparison_start_index
            in_batch_pairs_within_threshold = list(
                zip(anchor_indices, comparison_indices)
            )
            pairs_within_threshold.extend(
                [(i, j) for (i, j) in in_batch_pairs_within_threshold if i < j]
            )

        return set(pairs_within_threshold)
