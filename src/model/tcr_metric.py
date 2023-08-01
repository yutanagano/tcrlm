from abc import abstractmethod, ABC
from numpy import ndarray
from typing import Iterable

from src.tcr import Tcr


class TcrMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def calc_distance_between(self, anchor: Tcr, comparison: Tcr) -> float:
        pass

    @abstractmethod
    def calc_cdist_matrix(self, anchors: Iterable[Tcr], comparisons: Iterable[Tcr]) -> ndarray:
        pass

    @abstractmethod
    def calc_pdist_vector(self, tcrs: Iterable[Tcr]) -> ndarray:
        pass