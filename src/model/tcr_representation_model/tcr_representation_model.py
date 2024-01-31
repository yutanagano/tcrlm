from abc import abstractmethod
from numpy import ndarray
from pandas import DataFrame
from scipy.spatial import distance
from src.model.tcr_metric import TcrMetric


class TcrRepresentationModel(TcrMetric):
    """
    See TcrMetric for specs of input DataFrames.
    """

    @property
    @abstractmethod
    def d_model(self) -> int:
        pass

    def calc_cdist_matrix(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame
    ) -> ndarray:
        anchor_tcr_representations = self.calc_vector_representations(anchor_tcrs)
        comparison_tcr_representations = self.calc_vector_representations(
            comparison_tcrs
        )
        return self.calc_cdist_matrix_from_representations(
            anchor_tcr_representations, comparison_tcr_representations
        )

    def calc_pdist_vector(self, tcrs: DataFrame) -> ndarray:
        tcr_representations = self.calc_vector_representations(tcrs)
        return self.calc_pdist_vector_from_representations(tcr_representations)

    @abstractmethod
    def calc_vector_representations(self, tcrs: DataFrame) -> ndarray:
        pass

    def calc_cdist_matrix_from_representations(
        self,
        anchor_tcr_representations: ndarray,
        comparison_tcr_representations: ndarray,
    ) -> ndarray:
        return distance.cdist(
            anchor_tcr_representations,
            comparison_tcr_representations,
            metric="euclidean",
        )

    def calc_pdist_vector_from_representations(
        self, tcr_representations: ndarray
    ) -> ndarray:
        return distance.pdist(tcr_representations, metric="euclidean")
