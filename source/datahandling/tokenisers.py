'''
Custom tokeniser classes for amino acid tokenisation.

Tokenisers should understand the following characters:

The mask character: ?
The amino acids:    ACDEFGHIKLMNPQRSTVWY

Because of how this codebase modularises tokenisation, the vocabulary can
change depending on the setting. To keep some consistency, there are two tokens
that should always be mapped to the same index:

- The padding token, which should always be mapped to 0
- The 'masked' or 'unknown' token, which should always be mapped to 1
'''


from itertools import product
import random
import torch
from typing import Iterable


amino_acids = (
    'A','C','D','E','F','G','H','I','K','L',
    'M','N','P','Q','R','S','T','V','W','Y'
)


class AaTokeniser:
    def __init__(self, len_tuplet: int) -> None:
        self._len_tuplet = len_tuplet
        self._token_to_int = dict()
        self._int_to_token = {
            0: ''
        }

        tuplet_vocabulary = sorted(
            [''.join(t) for t in product(amino_acids, repeat=len_tuplet)]
        )

        for i, tuplet in enumerate(tuplet_vocabulary):
            # Indices 0 and 1 are reserved for the padding and masking tokens
            self._token_to_int[tuplet] = 2+i
            self._int_to_token[2+i] = tuplet


    def tokenise(self, aa_seq: str) -> torch.Tensor:
        '''
        Turn an amino acid sequence from a raw string form to a tokenised
        tensor form.
        '''

        num_tokens = len(aa_seq) + 1 - self._len_tuplet
        tokens = []

        for i in range(num_tokens):
            token = aa_seq[i:i+self._len_tuplet]

            # Anything with an unknown residue is mapped to the mask token (1)
            if '?' in token:
                tokens.append(1)
                continue

            tokens.append(self._token_to_int[token])

        return torch.tensor(tokens, dtype=torch.long, requires_grad=False)


    def generate_mlm_pair(
        self,
        aa_seq: str,
        p_mask: float,
        p_mask_random: float,
        p_mask_keep: float
    ) -> tuple:
        seq_len = len(aa_seq)

        indices_to_mask = self._pick_masking_indices(
            seq_len=seq_len,
            p_mask=p_mask
        )
        masked_seq = self._generate_masked_seq(
            seq=aa_seq,
            indices_to_mask=indices_to_mask,
            p_mask_random=p_mask_random,
            p_mask_keep=p_mask_keep
        )
        tokenised_x = self.tokenise(aa_seq=masked_seq)

        adjusted_indices = self._adjust_indices(
            indices=indices_to_mask,
            seq_len=seq_len
        )
        tokenised_seq = self.tokenise(aa_seq=aa_seq)
        tokenised_y = self._apply_index_filter(
            tokenised_seq=tokenised_seq,
            indices=adjusted_indices
        )

        return (tokenised_x, tokenised_y)


    def _pick_masking_indices(self, seq_len: int, p_mask: float) -> list:
        if p_mask < 0 or p_mask >= 1:
            raise RuntimeError(f'p_mask must lie in [0,1): {p_mask}')

        if p_mask == 0:
            return []
        
        num_to_be_masked = max(1, round(seq_len * p_mask))
        return random.sample(range(seq_len), num_to_be_masked)


    def _generate_masked_seq(
        self,
        seq: str,
        indices_to_mask: Iterable,
        p_mask_random: float,
        p_mask_keep: float
    ) -> str:
        if p_mask_random < 0 or p_mask_random > 1:
            raise RuntimeError(
                f'p_mask_random must lie in [0,1]: {p_mask_random}'
            )

        if p_mask_keep < 0 or p_mask_keep > 1:
            raise RuntimeError(
                f'p_mask_keep must lie in [0,1]: {p_mask_keep}'
            )

        if p_mask_random + p_mask_keep > 1:
            raise RuntimeError(
                'p_mask_random + p_mask_keep must be less than 1.'
            )

        seq = list(seq)

        for i in indices_to_mask:
            r = random.random()
            if r < p_mask_random:
                # Ensure residue is replaced with a DISTINCT residue
                seq[i] = random.sample(set(amino_acids) - {seq[i]},1)[0]
                continue

            if r < 1 - p_mask_keep:
                seq[i] = '?'
                continue
        
        return ''.join(seq)


    def _adjust_indices(self, indices: Iterable, seq_len: int) -> set:
        adjusted_indices = []

        for i in indices:
            adjusted = [
                new_i for new_i in range(i+1-self._len_tuplet, i+1) \
                if new_i >= 0 and new_i < seq_len + 1 - self._len_tuplet
            ]

            adjusted_indices.extend(adjusted)

        return set(adjusted_indices)


    def _apply_index_filter(
        self,
        tokenised_seq: torch.Tensor,
        indices: Iterable
    ) -> torch.Tensor:
        filtered = torch.zeros_like(tokenised_seq)
        for i in indices:
            filtered[i] = tokenised_seq[i]
        return filtered