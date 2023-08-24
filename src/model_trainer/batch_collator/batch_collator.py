from abc import ABC, abstractmethod
from torch import LongTensor
from typing import Iterable, Tuple

from src.data.tokeniser.tokeniser import Tokeniser


class BatchCollator(ABC):
    def __init__(self, tokeniser: Tokeniser) -> None:
        self._token_vocabulary_index = tokeniser.token_vocabulary_index

    @abstractmethod
    def collate_fn(self, tokenised_tcrs: Iterable[LongTensor]) -> Tuple[LongTensor]:
        pass
