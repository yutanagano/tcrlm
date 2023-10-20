import itertools
import numpy as np
from pandas import DataFrame
from src.clustering import ClusteringMachine
from typing import Set


class BruteForceClusteringMachine(ClusteringMachine):
    def cluster(self, tcrs: DataFrame, distance_threshold: float) -> Set:
        tcr_indices = range(len(tcrs))
        possible_pairs = list(itertools.combinations(tcr_indices, 2))

        pdist_vector = self._tcr_representation_model.calc_pdist_vector(tcrs)
        (indices_of_pairs_within_threshold,) = np.nonzero(pdist_vector <= distance_threshold)
        pairs_within_threshold = [possible_pairs[index] for index in indices_of_pairs_within_threshold]

        return set(pairs_within_threshold)
