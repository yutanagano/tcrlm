from abc import ABC, abstractmethod
from torch import Tensor

from src.tcr import Tcr


class Tokeniser(ABC):
    @abstractmethod
    def tokenise(self, tcr: Tcr) -> Tensor:
        pass

    @abstractmethod
    def tokenise_with_dropout(self, tcr: Tcr) -> Tensor:
        pass