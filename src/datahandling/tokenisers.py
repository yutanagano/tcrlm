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
import random
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
    def tokenise(self, tcr: Series, noising: bool = False) -> Tensor:
        '''
        Given a pandas Series containing information on a particular TCR,
        generate a tokenisation of it. If noising, then randomly drop out
        segments of the TCR representation.
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


class CDR3Tokeniser(_AATokeniser):
    '''
    Basic tokeniser which will tokenise a TCR in terms of its alpha and beta
    chain CDR3 amino acid sequences.
    '''


    def tokenise(self, tcr: Series) -> Tensor:
        '''
        Tokenise a TCR in terms of its alpha and/or beta chain CDR3 amino acid
        sequences.

        :return:
            Tensor with CDR3A/B residues.
            Dim 0 - token ID
            Dim 1 - token pos
            Dim 2 - compartment length
            Dim 3 - chain ID
        '''

        cdr3a = tcr.loc['CDR3A']
        cdr3b = tcr.loc['CDR3B']

        tokenised = [[2,0,0,0]]

        if notna(cdr3a):
            cdr3a_size = len(cdr3a)
            for i, aa in enumerate(cdr3a):
                tokenised.append([self._aa_to_index[aa], i+1, cdr3a_size, 1])

        if notna(cdr3b):
            cdr3b_size = len(cdr3b)
            for i, aa in enumerate(cdr3b):
                tokenised.append([self._aa_to_index[aa], i+1, cdr3b_size, 2])

        if len(tokenised) == 1:
            raise ValueError(f'No CDR3 data found in row {tcr.name}.')

        return torch.tensor(tokenised, dtype=torch.long)


class BCDR3Tokeniser(_AATokeniser):
    '''
    Basic tokeniser which will tokenise a TCR in terms of its beta chain CDR3.
    '''


    def tokenise(self, tcr: Series, noising: bool = False) -> Tensor:
        '''
        Tokenise a TCR in terms of its beta chain CDR3 amino acid sequences.

        :return:
            Tensor with CDR3B AA residues.
            Dim 0 - token ID
            Dim 1 - token pos
            Dim 2 - compartment length
        '''
    
        cdr3b = tcr.loc['CDR3B']

        tokenised = [[2,0,0]]

        if isna(cdr3b):
            raise ValueError(f'CDR3B data missing from row {tcr.name}')

        cdr3b_size = len(cdr3b)
        for i, aa in enumerate(cdr3b):
            if noising and random.random() < 0.2:
                continue

            tokenised.append([self._aa_to_index[aa], i+1, cdr3b_size])

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


    def tokenise(self, tcr: Series) -> Tensor:
        '''
        Tokenise a TCR in terms of its beta chain V gene and CDR3 amino acid
        sequence.

        :return:
            Tensor with beta V gene and CDR3 AA residues.
            Dim 0 - token ID
            Dim 1 - token pos
            Dim 2 - compartment length
            Dim 3 - compartment ID
        '''

        trbv = None if isna(tcr.loc['TRBV']) else tcr.loc['TRBV'].split('*')[0]
        cdr3b = tcr.loc['CDR3B']

        tokenised = [[2,0,0,0]]

        if notna(trbv):
            tokenised.append([self._v_to_index[trbv], 0, 0, 1])

        if notna(cdr3b):
            cdr3b_size = len(cdr3b)
            for i, aa in enumerate(cdr3b):
                tokenised.append([self._aa_to_index[aa], i+1, cdr3b_size, 2])
        
        if len(tokenised) == 1:
            raise ValueError(f'No TCRB data found in row {tcr.name}.')

        return torch.tensor(tokenised, dtype=torch.long)


class BCDRTokeniser(_AATokeniser):
    '''
    Tokeniser that takes the beta V gene and CDR3, and represents the chain as
    the set of CDRs 1, 2 and 3.
    '''


    def tokenise(self, tcr: Series) -> Tensor:
        '''
        Tokenise a TCR in terms of the amino acid sequences of its 3 CDRs.

        Amino acids get mapped as in the following: 'A' -> 3, 'C' -> 4, ...
        'Y' -> 22.

        :return:
            Tensor with beta CDR AA residues.
            Dim 0 - token ID
            Dim 1 - token pos
            Dim 2 - compartment length
            Dim 3 - compartment ID
        '''


        trbv = tcr.loc['TRBV']
        cdr3b = tcr.loc['CDR3B']

        cdr1b = None if isna(trbv) else V_CDRS[trbv]['CDR1-IMGT']
        cdr2b = None if isna(trbv) else V_CDRS[trbv]['CDR2-IMGT']

        tokenised = [[2,0,0,0]]

        if notna(cdr1b):
            cdr1b_size = len(cdr1b)
            for i, aa in enumerate(cdr1b):
                tokenised.append([self._aa_to_index[aa], i+1, cdr1b_size, 1])

        if notna(cdr2b):
            cdr2b_size = len(cdr2b)
            for i, aa in enumerate(cdr2b):
                tokenised.append([self._aa_to_index[aa], i+1, cdr2b_size, 2])

        if notna(cdr3b):
            cdr3b_size = len(cdr3b)
            for i, aa in enumerate(cdr3b):
                tokenised.append([self._aa_to_index[aa], i+1, cdr3b_size, 3])

        if len(tokenised) == 1:
            raise ValueError(f'No TCRB data found in row {tcr.name}.')

        return torch.tensor(tokenised, dtype=torch.long)