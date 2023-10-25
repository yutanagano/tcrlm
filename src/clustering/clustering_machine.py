from abc import ABC, abstractmethod
from pandas import DataFrame
from src.model.tcr_metric import TcrMetric
from typing import Set, Tuple


class ClusteringMachine(ABC):
    _tcr_metric: TcrMetric

    def __init__(self, tcr_metric: TcrMetric) -> None:
        self._tcr_metric = tcr_metric

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def cluster(self, tcrs: DataFrame, distance_threshold: float) -> Set[Tuple[int]]:
        pass
