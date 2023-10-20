from numpy import ndarray
from pandas import DataFrame
from scipy.spatial import KDTree
from src.model.tcr_representation_model import TcrRepresentationModel


class KdTreeClusteringMachine:
    _tcr_representation_model: TcrRepresentationModel

    def __init__(self, tcr_representation_model: TcrRepresentationModel) -> None:
        self._tcr_representation_model = tcr_representation_model

    def cluster_with_distance_threshold(self, tcrs: DataFrame, distance_threshold: float) -> ndarray:
        tcrs_as_vectors = self._tcr_representation_model.calc_vector_representations(tcrs)
        kd_tree = KDTree(data=tcrs_as_vectors, compact_nodes=True, balanced_tree=True)
        pairs_of_indices = kd_tree.query_pairs(r=distance_threshold, p=2)
        return pairs_of_indices
