from abc import ABC, abstractmethod
from enum import IntEnum
from torch import Tensor

from src.schema import Tcr


class Tokeniser(ABC):
    @property
    @abstractmethod
    def token_vocabulary_index(self) -> IntEnum:
        pass

    @abstractmethod
    def tokenise(self, tcr: Tcr) -> Tensor:
        pass
