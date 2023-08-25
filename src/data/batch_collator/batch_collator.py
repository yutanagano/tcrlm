from abc import ABC, abstractmethod
from torch import LongTensor
from typing import Iterable, Tuple

from src.data.tokeniser.tokeniser import Tokeniser
from src.data.tcr_pmhc_pair import TcrPmhcPair


class BatchCollator(ABC):
    def __init__(self, tokeniser: Tokeniser) -> None:
        self._tokeniser = tokeniser

    @abstractmethod
    def collate_fn(self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]) -> Tuple[LongTensor]:
        pass
