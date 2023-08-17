from abc import abstractmethod
from numpy import ndarray
from pandas import DataFrame
from scipy.spatial import distance

from src.model.tcr_metric.tcr_metric import TcrMetric


class TcrRepresentationModel(TcrMetric):
    """
    See TcrMetric for specs of input DataFrames.
    """

    def calc_cdist_matrix(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame
    ) -> ndarray:
        anchor_representations = self.calc_representations_of(anchor_tcrs)
        comparison_representations = self.calc_representations_of(comparison_tcrs)

        return distance.cdist(
            anchor_representations, comparison_representations, metric="euclidean"
        )

    def calc_pdist_vector(self, tcrs: DataFrame) -> ndarray:
        tcr_representations = self.calc_representations_of(tcrs)
        return distance.pdist(tcr_representations, metric="euclidean")

    @abstractmethod
    def calc_representations_of(self, tcrs: DataFrame) -> ndarray:
        pass
