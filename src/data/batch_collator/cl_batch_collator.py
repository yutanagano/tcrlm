from libtcrlm.schema import TcrPmhcPair, Tcr
from libtcrlm.tokeniser.token_indices import DefaultTokenIndex
from libtcrlm.tokeniser import Tokeniser
import numpy as np
import random
from src.data.batch_collator import MlmBatchCollator
from typing import Iterable, Tuple, List
import torch
from torch import BoolTensor, LongTensor, Tensor


class ClBatchCollator(MlmBatchCollator):
    PROB_DROP_ALPHA_GIVEN_DROP_CHAIN = 0.5

    def __init__(self, tokeniser: Tokeniser, frac_dropped_tokens: float = 0.2, prob_drop_chain: float = 0.5) -> None:
        super().__init__(tokeniser)
        self._frac_dropped_tokens = frac_dropped_tokens
        self._prob_drop_chain = prob_drop_chain

    def collate_fn(self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]) -> Tuple[Tensor]:
        double_view_batch = self._generate_double_view_batch(tcr_pmhc_pairs)
        double_view_positives_mask = self._generate_double_view_positives_mask(
            tcr_pmhc_pairs
        )
        masked_tcrs_padded, mlm_targets_padded = super().collate_fn(tcr_pmhc_pairs)

        return (
            double_view_batch,
            double_view_positives_mask,
            masked_tcrs_padded,
            mlm_targets_padded,
        )

    def _generate_double_view_batch(
        self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]
    ) -> LongTensor:
        tokenised_tcrs_view_1 = self._generate_tcr_view(tcr_pmhc_pairs)
        tokenised_tcrs_view_2 = self._generate_tcr_view(tcr_pmhc_pairs)
        double_view_batch = self._pad_tokenised_sequences(
            tokenised_tcrs_view_1 + tokenised_tcrs_view_2
        )
        return double_view_batch

    def _generate_tcr_view(self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]) -> LongTensor:
        tcrs_with_chains_maybe_dropped = [
            self._maybe_drop_chain(pair.tcr) for pair in tcr_pmhc_pairs
        ]
        tokenised_tcrs = [
            self._tokeniser.tokenise(tcr) for tcr in tcrs_with_chains_maybe_dropped
        ]
        censored_tcrs = self._randomly_drop_tokens(tokenised_tcrs)
        return censored_tcrs

    def _maybe_drop_chain(self, tcr: Tcr) -> Tcr:
        if not tcr.both_chains_specified or self._prob_drop_chain == 0:
            return tcr

        new_tcr = tcr.copy()

        will_drop_chain = random.random() < self._prob_drop_chain
        if will_drop_chain:
            will_drop_alpha = random.random() < self.PROB_DROP_ALPHA_GIVEN_DROP_CHAIN
            if will_drop_alpha:
                new_tcr.drop_tra()
            else:
                new_tcr.drop_trb()

        return new_tcr

    def _randomly_drop_tokens(self, tokenised_tcrs: LongTensor) -> LongTensor:
        if self._frac_dropped_tokens == 0:
            return tokenised_tcrs

        indices_of_tokens_to_drop = [
            self._choose_random_subset_of_indices(
                tcr, self._frac_dropped_tokens
            )
            for tcr in tokenised_tcrs
        ]
        censored_tcrs = [
            self._drop_tokens_at_indices(tcr, indices_to_drop)
            for tcr, indices_to_drop in zip(tokenised_tcrs, indices_of_tokens_to_drop)
        ]
        return censored_tcrs

    def _drop_tokens_at_indices(
        self, tokenised_tcr: LongTensor, indices_to_drop: List[int]
    ) -> LongTensor:
        censored_tcr = tokenised_tcr.clone()
        TOKEN_ID_DIM = 0

        for idx_to_drop in indices_to_drop:
            censored_tcr[idx_to_drop, TOKEN_ID_DIM] = DefaultTokenIndex.NULL

        return censored_tcr

    def _generate_double_view_positives_mask(
        self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]
    ) -> BoolTensor:
        num_samples_in_single_view = len(tcr_pmhc_pairs)
        single_view_identities = np.eye(num_samples_in_single_view)

        pmhc_array = np.array([pair.pmhc for pair in tcr_pmhc_pairs])
        single_view_pmhc_matches = (
            pmhc_array[:, np.newaxis] == pmhc_array[np.newaxis, :]
        )

        single_view_positives_including_identities = np.logical_or(
            single_view_identities, single_view_pmhc_matches
        )

        double_view_identities = np.eye(2 * num_samples_in_single_view)
        double_view_positives_including_identities = np.tile(
            single_view_positives_including_identities, reps=(2, 2)
        )
        double_view_positives = np.logical_and(
            double_view_positives_including_identities,
            np.logical_not(double_view_identities),
        )

        return torch.tensor(double_view_positives, dtype=torch.bool)
