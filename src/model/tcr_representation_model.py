from abc import abstractmethod
import numpy as np
from numpy import ndarray
from scipy.spatial import distance
from typing import Iterable

from src.tcr import Tcr
from src.model.tcr_metric import TcrMetric


class TcrRepresentationModel(TcrMetric):
    def calc_distance_between(self, anchor: Tcr, comparison: Tcr) -> float:
        anchor_representation = self.calc_representation_of(anchor)
        comparison_representation = self.calc_representation_of(comparison)
        difference = anchor_representation - comparison_representation

        return np.linalg.norm(difference, ord=2)
    
    def calc_cdist_matrix(self, anchors: Iterable[Tcr], comparisons: Iterable[Tcr]) -> ndarray:
        anchor_representations = self.calc_representations_of(anchors)
        comparison_representations = self.calc_representations_of(comparisons)
        
        return distance.cdist(anchor_representations, comparison_representations, metric="euclidean")
    
    def calc_pdist_vector(self, tcrs: Iterable[Tcr]) -> ndarray:
        tcr_representations = self.calc_representations_of(tcrs)
        return distance.pdist(tcr_representations, metric="euclidean")

    def calc_representation_of(self, tcr: Tcr) -> ndarray:
        return self.calc_representations_of([tcr])[0]

    @abstractmethod
    def calc_representations_of(self, tcrs: Iterable[Tcr]) -> ndarray:
        pass