from typing import Iterable, Tuple
from torch import Tensor
from src.nn.data.batch_collator import BatchCollator
from src.schema import TcrPmhcPair


class NoOpCollator(BatchCollator):
    def collate_fn(self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]) -> Tuple[Tensor]:
        return tcr_pmhc_pairs
