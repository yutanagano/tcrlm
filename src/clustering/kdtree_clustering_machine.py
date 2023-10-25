from pandas import DataFrame
from scipy.spatial import KDTree
from src.clustering import ClusteringMachine
from src.model.tcr_metric import TcrMetric
from src.model.tcr_representation_model import TcrRepresentationModel
from typing import Set


class KdTreeClusteringMachine(ClusteringMachine):
    def __init__(self, tcr_metric: TcrMetric) -> None:
        if not isinstance(tcr_metric, TcrRepresentationModel):
            raise RuntimeError("tcr_metric must be TcrRepresentationModel")

        super().__init__(tcr_metric)

    @property
    def name(self) -> str:
        return f"{self._tcr_metric.name} (KD Tree)"

    def cluster(self, tcrs: DataFrame, distance_threshold: float) -> Set:
        tcrs_as_vectors = self._tcr_metric.calc_vector_representations(tcrs)
        kd_tree = KDTree(data=tcrs_as_vectors, compact_nodes=True, balanced_tree=True)
        pairs_of_indices = kd_tree.query_pairs(r=distance_threshold, p=2)
        return pairs_of_indices
