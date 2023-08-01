from abc import abstractmethod, ABC
from numpy import ndarray
from typing import Iterable

from src.tcr import Tcr


class TcrRepresentationModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def calc_representation_of(self, tcr: Tcr) -> ndarray:
        pass

    @abstractmethod
    def calc_representations_of(self, tcrs: Iterable[Tcr]) -> ndarray:
        pass