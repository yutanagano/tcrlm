'''
Custom tokeniser classes for TCR tokenisation.

Because of how this codebase modularises tokenisation, the vocabulary can
change depending on the setting. To keep some consistency, there is one index
that is reserved in all cases for padding values:

0: reserved for <pad>

In addition, in the case of token indices (as opposed to indices that may
describe chain, position, etc.), there are two more reserved indices:

1: reserved for <mask>
2: reserved for <cls>
'''


from abc import ABC, abstractmethod
from pandas import isna, notna, Series
import torch
from torch import Tensor


amino_acids = (
    'A','C','D','E','F','G','H','I','K','L',
    'M','N','P','Q','R','S','T','V','W','Y'
)


class Tokeniser(ABC):
    '''
    Abstract base class for tokenisers.
    '''

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        '''
        Return this tokeniser's vocabulary size.
        '''


    @abstractmethod
    def tokenise(self, tcr: Series, chain: str = 'both') -> Tensor:
        '''
        Given a pandas Series containing information on a particular TCR,
        generate a tokenisation of it.
        '''


class CDR3Tokeniser(Tokeniser):
    '''
    Basic tokeniser which will tokenise a TCR in terms of its alpha and beta
    chain CDR3 amino acid sequences.
    '''

    def __init__(self) -> None:
        self._aa_to_index = dict()

        for i, aa in enumerate(amino_acids):
            self._aa_to_index[aa] = 3+i # offset for reserved tokens


    @property
    def vocab_size(self) -> int:
        return 20


    def tokenise(self, tcr: Series, chain: str = 'both') -> Tensor:
        '''
        Tokenise a TCR in terms of its alpha and/or beta chain CDR3 amino acid
        sequences.

        Amino acids get mapped as in the following: 'A' -> 3, 'C' -> 4, ...
        'Y' -> 22.

        :return: Tensor where every column represents an amino acid residue
            from either the alpha or beta CDR3s, except the first column is
            always a <cls> token. Each column is a 3-dimensional vector where
            the first element is the amino acid / token index (as described
            above), the second element is an integer indicating whether the
            residue came from the alpha (1) or beta (2) CDR3, and the third
            element is an integer indicating the residue position within its
            chain (1-indexed). Depending on the input for `chain`, either both,
            only alpha, or only the beta chain will be tokenised, unless the
            specified chain is not available, in which case it will default
            back to tokenising the available chain.
        '''
        if not chain in ('both', 'alpha', 'beta'):
            raise ValueError(f'Unrecognised value for chain: {chain}')

        cdr3a = tcr.loc['CDR3A']
        cdr3b = tcr.loc['CDR3B']

        tokenised = [[2,0,0]]

        if isna(cdr3a) and chain == 'alpha':
            chain = 'beta'

        if isna(cdr3b) and chain == 'beta':
            chain = 'alpha'

        if notna(cdr3a) and chain in ('both', 'alpha'):
            for i, aa in enumerate(cdr3a):
                tokenised.append([self._aa_to_index[aa], 1, i+1])

        if notna(cdr3b) and chain in ('both', 'beta'):
            for i, aa in enumerate(cdr3b):
                tokenised.append([self._aa_to_index[aa], 2, i+1])

        return torch.tensor(tokenised, dtype=torch.long)