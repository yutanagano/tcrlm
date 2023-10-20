from abc import ABC, abstractmethod
from pandas import DataFrame
from src.model.tcr_representation_model import TcrRepresentationModel
from typing import Set, Tuple


class ClusteringMachine(ABC):
    _tcr_representation_model: TcrRepresentationModel

    def __init__(self, tcr_representation_model: TcrRepresentationModel) -> None:
        self._tcr_representation_model = tcr_representation_model

    @abstractmethod
    def cluster(self, tcrs: DataFrame, distance_threshold: float) -> Set[Tuple[int]]:
        pass
