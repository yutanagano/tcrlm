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
from src.resources import *
import torch
from torch import Tensor


class _Tokeniser(ABC):
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


class _AATokeniser(_Tokeniser):
    '''
    Base class for tokenisers focusing on AA-level tokenisation.
    '''


    def __init__(self) -> None:
        self._aa_to_index = dict()

        for i, aa in enumerate(AMINO_ACIDS):
            self._aa_to_index[aa] = 3+i # offset for reserved tokens


    @property
    def vocab_size(self) -> int:
        return 20


class ABCDR3Tokeniser(_AATokeniser):
    '''
    Basic tokeniser which will tokenise a TCR in terms of its alpha and beta
    chain CDR3 amino acid sequences.
    '''


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
            above), the second element is an integer indicating the residue
            position within its chain (1-indexed), and the third element is an
            integer indicating whether the residue came from the alpha (1) or
            beta (2) CDR3. Depending on the input for `chain`, either both,
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
                tokenised.append([self._aa_to_index[aa], i+1, 1])

        if notna(cdr3b) and chain in ('both', 'beta'):
            for i, aa in enumerate(cdr3b):
                tokenised.append([self._aa_to_index[aa], i+1, 2])

        if len(tokenised) == 1:
            raise ValueError(f'No CDR3 data found in row {tcr.name}.')

        return torch.tensor(tokenised, dtype=torch.long)


class BCDR3Tokeniser(_AATokeniser):
    '''
    Basic tokeniser which will tokenise a TCR in terms of its beta chain CDR3.
    '''


    def tokenise(self, tcr: Series, chain: str = 'both') -> Tensor:
        '''
        Tokenise a TCR in terms of its beta chain CDR3 amino acid sequences.

        Amino acids get mapped as in the following: 'A' -> 3, 'C' -> 4, ...
        'Y' -> 22.

        :return: Tensor where every column represents an amino acid residue
            from the beta CDR3s, except the first column is always a <cls>
            token. Each column is a 2-dimensional vector where the first
            element is the amino acid / token index (as described above), the
            second element is an integer indicating the residue position within
            its chain (1-indexed).
        '''
    
        cdr3b = tcr.loc['CDR3B']

        tokenised = [[2,0]]

        if isna(cdr3b):
            raise ValueError(f'CDR3B data missing from row {tcr.name}')

        for i, aa in enumerate(cdr3b):
            tokenised.append([self._aa_to_index[aa], i+1])

        return torch.tensor(tokenised, dtype=torch.long)
    

class BVCDR3Tokeniser(_Tokeniser):
    '''
    Tokeniser which takes the beta chain V gene and CDR3 sequence.
    '''

    def __init__(self) -> None:
        self._aa_to_index = dict()
        self._v_to_index = dict()


        for i, aa in enumerate(AMINO_ACIDS):
            self._aa_to_index[aa] = 3+i # offset for reserved tokens

        for i, trbv in enumerate(FUNCTIONAL_TRBVS):
            self._v_to_index[trbv] = 3+20+i # offset for reserved tokens and amino acid tokens


    @property
    def vocab_size(self) -> int:
        return 20 + 48 #aas + trbvs


    def tokenise(self, tcr: Series, chain: str = 'both') -> Tensor:
        '''
        Tokenise a TCR in terms of its beta chain V gene and CDR3 amino acid
        sequence.

        Amino acids get mapped as in the following: 'A' -> 3, 'C' -> 4, ...
        'Y' -> 22.

        TRBV genes get mapped as in the following: 'TRBV2' -> 23, ...
        'TRBV30' -> 70.

        :return: Tensor where the first column is always a <cls>
            token, followed by a v gene token, then tokens representing amino
            acids from the CDR3B. Each column is a 3-dimensional vector where
            the first element is the amino acid / token index (as described
            above), the second element is an integer indicating the residue
            position within its chain (1-indexed), and the third element is an
            integer indicating whether the token represents something from the
            CDR3, or a V gene.
        '''

        trbv = None if isna(tcr.loc['TRBV']) else tcr.loc['TRBV'].split('*')[0]
        cdr3b = tcr.loc['CDR3B']

        tokenised = [[2,0,0]]

        if notna(trbv):
            tokenised.append([self._v_to_index[trbv], 0, 1])

        if notna(cdr3b):
            for i, aa in enumerate(cdr3b):
                tokenised.append([self._aa_to_index[aa], i+1, 2])
        
        if len(tokenised) == 1:
            raise ValueError(f'No TCRB data found in row {tcr.name}.')

        return torch.tensor(tokenised, dtype=torch.long)


class BCDRTokeniser(_AATokeniser):
    '''
    Tokeniser that takes the beta V gene and CDR3, and represents the chain as
    the set of CDRs 1, 2 and 3.
    '''


    def tokenise(self, tcr: Series, chain: str = 'both') -> Tensor:
        '''
        Tokenise a TCR in terms of the amino acid sequences of its 3 CDRs.

        Amino acids get mapped as in the following: 'A' -> 3, 'C' -> 4, ...
        'Y' -> 22.

        :return:
            Tensor where the first column is always a <cls> token, followed by
            tokens representing amino acids from the CDR1, then CDR2, then CDR3
            regions. Each column is a 3-dimensional vector where the first
            element is the amino acid index, the second element is an integer
            indicating the residue position within its compartment (1-indexed),
            and the third element is an integer indicating whether the token
            represents something from the CDR1, 2, or 3.
        '''


        trbv = tcr.loc['TRBV']
        cdr3b = tcr.loc['CDR3B']

        cdr1b = None if isna(trbv) else V_CDRS[trbv]['CDR1-IMGT']
        cdr2b = None if isna(trbv) else V_CDRS[trbv]['CDR2-IMGT']

        tokenised = [[2,0,0]]

        if notna(cdr1b):
            for i, aa in enumerate(cdr1b):
                tokenised.append([self._aa_to_index[aa], i+1, 1])

        if notna(cdr2b):
            for i, aa in enumerate(cdr2b):
                tokenised.append([self._aa_to_index[aa], i+1, 2])

        if notna(cdr3b):
            for i, aa in enumerate(cdr3b):
                tokenised.append([self._aa_to_index[aa], i+1, 3])

        if len(tokenised) == 1:
            raise ValueError(f'No TCRB data found in row {tcr.name}.')

        return torch.tensor(tokenised, dtype=torch.long)