from typing import Iterable, Tuple
from torch import LongTensor
from torch.nn import utils

from src.nn.data.batch_collator import BatchCollator
from src.nn.data.tokeniser.token_indices import DefaultTokenIndex
from src.nn.data.schema.tcr_pmhc_pair import TcrPmhcPair


class DefaultBatchCollator(BatchCollator):
    def collate_fn(self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]) -> Tuple[LongTensor]:
        batch = [
            self._tokeniser.tokenise(tcr_pmhc_pair.tcr)
            for tcr_pmhc_pair in tcr_pmhc_pairs
        ]
        padded_batch = utils.rnn.pad_sequence(
            sequences=batch, batch_first=True, padding_value=DefaultTokenIndex.NULL
        )
        return (padded_batch,)
