from abc import ABC, abstractmethod
from libtcrlm.tokeniser import Tokeniser
from libtcrlm.schema import TcrPmhcPair
from torch import Tensor
from typing import Iterable, Tuple


class BatchCollator(ABC):
    def __init__(self, tokeniser: Tokeniser) -> None:
        self._tokeniser = tokeniser

    @abstractmethod
    def collate_fn(self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]) -> Tuple[Tensor]:
        pass
