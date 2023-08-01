from pandas import Series
from torch import Tensor
from torch.utils.data import Dataset
from typing import Iterable

from src.tcr import Tcr
from src.model.data.tokeniser.tokeniser import Tokeniser


class TcrDataset(Dataset):
    def __init__(self, data: Iterable[Tcr], tokeniser: Tokeniser):
        super().__init__()
        self._tcr_series = self._generate_tcr_series_from(data)
        self._tokeniser = tokeniser

    def __len__(self) -> int:
        return len(self._tcr_series)

    def __getitem__(self, index: int) -> Tensor:
        tcr_at_index = self._tcr_series.iloc[index]
        return self._tokenise(tcr_at_index)

    def _generate_tcr_series_from(data: Iterable[Tcr]) -> Series:
        tcr_series = Series(data)
        return tcr_series
    
    def _tokenise(self, tcr: Tcr) -> Tensor:
        return self._tokeniser.tokenise(tcr)