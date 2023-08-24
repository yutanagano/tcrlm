from abc import abstractmethod
from numpy import ndarray
from pandas import DataFrame

from src.model.tcr_metric import TcrMetric


class TcrRepresentationModel(TcrMetric):
    """
    See TcrMetric for specs of input DataFrames.
    """

    @abstractmethod
    def calc_vector_representations(self, tcrs: DataFrame) -> ndarray:
        pass
