from typing import Iterable, Tuple
from torch import Tensor
from src.nn.data.batch_collator import BatchCollator
from src.nn.data.schema.tcr_pmhc_pair import TcrPmhcPair


class NoOpCollator(BatchCollator):
    def collate_fn(self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]) -> Tuple[Tensor]:
        return tcr_pmhc_pairs