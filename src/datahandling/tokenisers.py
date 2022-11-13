'''
Custom tokeniser classes for TCR tokenisation.

Because of how this codebase modularises tokenisation, the vocabulary can
change depending on the setting. To keep some consistency, there are two
indices that are reserved for the following special tokens:

0: reserved for the padding token
1: reserved for the 'masked' or 'unknown' token
'''


from abc import ABC, abstractmethod
from pandas import notna, Series
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

    @abstractmethod
    def tokenise(self, tcr: Series) -> Tensor:
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
            self._aa_to_index[aa] = 2+i


    def tokenise(self, tcr: Series) -> Tensor:
        '''
        Tokenise a TCR in terms of its alpha and beta chain CDR3 amino acid
        sequences.

        Amino acids get mapped as in the following: 'A' -> 2, 'C' -> 3, ...
        'Y' -> 21.

        :return: Tensor where every column represents an amino acid residue
            from either the alpha or beta CDR3s. Each column is a 2-dimensional
            vector where the first element is the amino acid index (as
            described above) and the second element is an integer indicating
            whether the residue came from the alpha (0) or beta (1) CDR3.
        '''

        cdr3a = tcr.loc['CDR3A']
        cdr3b = tcr.loc['CDR3B']

        tokenised = []

        if notna(cdr3a):
            for aa in cdr3a:
                tokenised.append([self._aa_to_index[aa], 0])

        if notna(cdr3b):
            for aa in cdr3b:
                tokenised.append([self._aa_to_index[aa], 1])

        return torch.tensor(tokenised, dtype=torch.long)