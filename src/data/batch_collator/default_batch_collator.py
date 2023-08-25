from typing import Iterable, Tuple
from torch import LongTensor
from torch.nn import utils

from src.data.batch_collator import BatchCollator
from src.data.tokeniser.token_indices import DefaultTokenIndex
from src.data.tcr_pmhc_pair import TcrPmhcPair


class DefaultBatchCollator(BatchCollator):
    def collate_fn(self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]) -> Tuple[LongTensor]:
        batch = [self._tokeniser.tokenise(tcr_pmhc_pair.tcr) for tcr_pmhc_pair in tcr_pmhc_pairs]
        return utils.rnn.pad_sequence(
            sequences=batch, batch_first=True, padding_value=DefaultTokenIndex.NULL
        )