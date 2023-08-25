from typing import Iterable, Tuple, List
from torch import LongTensor

from src.data.batch_collator import MlmBatchCollator
from src.data.tokeniser.token_indices import DefaultTokenIndex
from src.data.tcr_pmhc_pair import TcrPmhcPair


class AclBatchCollator(MlmBatchCollator):
    PROPORTION_OF_TOKENS_TO_CENSOR = 0.2

    def collate_fn(self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]) -> Tuple[LongTensor]:
        tokenised_tcrs = [self._tokeniser.tokenise(pair.tcr) for pair in tcr_pmhc_pairs]

        indices_of_tokens_to_drop_anchor_tcrs = [
            self._choose_random_subset_of_indices(
                tcr, self.PROPORTION_OF_TOKENS_TO_CENSOR
            )
            for tcr in tokenised_tcrs
        ]
        indices_of_tokens_to_drop_on_positive_pair_tcrs = [
            self._choose_random_subset_of_indices(
                tcr, self.PROPORTION_OF_TOKENS_TO_CENSOR
            )
            for tcr in tokenised_tcrs
        ]

        anchor_tcrs = [
            self._censor_tcr(tcr, indices_to_drop)
            for tcr, indices_to_drop in zip(
                tokenised_tcrs, indices_of_tokens_to_drop_anchor_tcrs
            )
        ]
        positive_pair_tcrs = [
            self._censor_tcr(tcr, indices_to_drop)
            for tcr, indices_to_drop in zip(
                tokenised_tcrs, indices_of_tokens_to_drop_on_positive_pair_tcrs
            )
        ]

        anchor_tcrs_padded = self._pad_tokenised_sequences(anchor_tcrs)
        positive_pair_tcrs_padded = self._pad_tokenised_sequences(positive_pair_tcrs)

        masked_tcrs_padded, mlm_targets_padded = super().collate_fn(tcr_pmhc_pairs)

        return (
            anchor_tcrs_padded,
            positive_pair_tcrs_padded,
            masked_tcrs_padded,
            mlm_targets_padded,
        )

    def _censor_tcr(
        self, tokenised_tcr: LongTensor, indices_to_drop: List[int]
    ) -> LongTensor:
        censored_tcr = tokenised_tcr.clone()
        TOKEN_ID_DIM = 0

        for idx_to_drop in indices_to_drop:
            censored_tcr[idx_to_drop, TOKEN_ID_DIM] = DefaultTokenIndex.NULL

        return censored_tcr
