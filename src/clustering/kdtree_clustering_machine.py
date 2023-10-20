from pandas import DataFrame
from scipy.spatial import KDTree
from src.clustering import ClusteringMachine
from typing import Set


class KdTreeClusteringMachine(ClusteringMachine):
    def cluster(self, tcrs: DataFrame, distance_threshold: float) -> Set:
        tcrs_as_vectors = self._tcr_representation_model.calc_vector_representations(tcrs)
        kd_tree = KDTree(data=tcrs_as_vectors, compact_nodes=True, balanced_tree=True)
        pairs_of_indices = kd_tree.query_pairs(r=distance_threshold, p=2)
        return pairs_of_indices
