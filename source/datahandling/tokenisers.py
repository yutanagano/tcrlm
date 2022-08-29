'Custom tokeniser classes for amino acid tokenisation.'


from itertools import product
import torch


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

        num_tokens = len(aa_seq) - self._len_tuplet + 1
        tokens = []

        for i in range(num_tokens):
            token = aa_seq[i:i+self._len_tuplet]

            # Anything with an unknown residue is mapped to the mask token (1)
            if '?' in token:
                tokens.append(1)
                continue

            tokens.append(self._token_to_int[token])

        return torch.tensor(tokens, dtype=torch.long, requires_grad=False)