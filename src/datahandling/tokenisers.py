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

travs = (
    "TRAV1-1",
    "TRAV1-2",
    "TRAV2",
    "TRAV3",
    "TRAV4",
    "TRAV5",
    "TRAV6",
    "TRAV7",
    "TRAV8-1",
    "TRAV8-2",
    "TRAV8-3",
    "TRAV8-4",
    "TRAV8-6",
    "TRAV9-1",
    "TRAV9-2",
    "TRAV10",
    "TRAV12-1",
    "TRAV12-2",
    "TRAV12-3",
    "TRAV13-1",
    "TRAV13-2",
    "TRAV14/DV4",
    "TRAV16",
    "TRAV17",
    "TRAV18",
    "TRAV19",
    "TRAV20",
    "TRAV21",
    "TRAV22",
    "TRAV23/DV6",
    "TRAV24",
    "TRAV25",
    "TRAV26-1",
    "TRAV26-2",
    "TRAV27",
    "TRAV29/DV5",
    "TRAV30",
    "TRAV34",
    "TRAV35",
    "TRAV36/DV7",
    "TRAV38-1",
    "TRAV38-2/DV8",
    "TRAV39",
    "TRAV40",
    "TRAV41"
)

trajs = (
    "TRAJ3",
    "TRAJ4",
    "TRAJ5",
    "TRAJ6",
    "TRAJ7",
    "TRAJ8",
    "TRAJ9",
    "TRAJ10",
    "TRAJ11",
    "TRAJ12",
    "TRAJ13",
    "TRAJ14",
    "TRAJ15",
    "TRAJ16",
    "TRAJ17",
    "TRAJ18",
    "TRAJ20",
    "TRAJ21",
    "TRAJ22",
    "TRAJ23",
    "TRAJ24",
    "TRAJ26",
    "TRAJ27",
    "TRAJ28",
    "TRAJ29",
    "TRAJ30",
    "TRAJ31",
    "TRAJ32",
    "TRAJ33",
    "TRAJ34",
    "TRAJ35",
    "TRAJ36",
    "TRAJ37",
    "TRAJ38",
    "TRAJ39",
    "TRAJ40",
    "TRAJ41",
    "TRAJ42",
    "TRAJ43",
    "TRAJ44",
    "TRAJ45",
    "TRAJ46",
    "TRAJ47",
    "TRAJ48",
    "TRAJ49",
    "TRAJ50",
    "TRAJ52",
    "TRAJ53",
    "TRAJ54",
    "TRAJ56",
    "TRAJ57"
)

trbvs = (
    "TRBV2",
    "TRBV3-1",
    "TRBV4-1",
    "TRBV4-2",
    "TRBV4-3",
    "TRBV5-1",
    "TRBV5-4",
    "TRBV5-5",
    "TRBV5-6",
    "TRBV5-8",
    "TRBV6-1",
    "TRBV6-2",
    "TRBV6-3",
    "TRBV6-4",
    "TRBV6-5",
    "TRBV6-6",
    "TRBV6-8",
    "TRBV6-9",
    "TRBV7-2",
    "TRBV7-3",
    "TRBV7-4",
    "TRBV7-6",
    "TRBV7-7",
    "TRBV7-8",
    "TRBV7-9",
    "TRBV9",
    "TRBV10-1",
    "TRBV10-2",
    "TRBV10-3",
    "TRBV11-1",
    "TRBV11-2",
    "TRBV11-3",
    "TRBV12-3",
    "TRBV12-4",
    "TRBV12-5",
    "TRBV13",
    "TRBV14",
    "TRBV15",
    "TRBV16",
    "TRBV18",
    "TRBV19",
    "TRBV20-1",
    "TRBV24-1",
    "TRBV25-1",
    "TRBV27",
    "TRBV28",
    "TRBV29-1",
    "TRBV30"
)


trbjs = (
    "TRBJ1-1",
    "TRBJ1-2",
    "TRBJ1-3",
    "TRBJ1-4",
    "TRBJ1-5",
    "TRBJ1-6",
    "TRBJ2-1",
    "TRBJ2-2",
    "TRBJ2-3",
    "TRBJ2-4",
    "TRBJ2-5",
    "TRBJ2-6",
    "TRBJ2-7"
)


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


class _CDR3Tokeniser(_Tokeniser):
    '''
    Base class for CDR3 tokenisers.
    '''


    def __init__(self) -> None:
        self._aa_to_index = dict()

        for i, aa in enumerate(amino_acids):
            self._aa_to_index[aa] = 3+i # offset for reserved tokens


    @property
    def vocab_size(self) -> int:
        return 20


class ABCDR3Tokeniser(_CDR3Tokeniser):
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

        return torch.tensor(tokenised, dtype=torch.long)


class BCDR3Tokeniser(_CDR3Tokeniser):
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


        for i, aa in enumerate(amino_acids):
            self._aa_to_index[aa] = 3+i # offset for reserved tokens

        for i, trbv in enumerate(trbvs):
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
        trbv = tcr.loc['TRBV']
        cdr3b = tcr.loc['CDR3B']

        tokenised = [[2,0,0]]

        if isna(cdr3b) or isna(trbv):
            raise ValueError(f'CDR3 data missing from row {tcr.index}')

        tokenised.append([self._v_to_index[trbv]], 0, 1)

        for i, aa in enumerate(cdr3b):
            tokenised.append([self._aa_to_index[aa], i+1, 2])

        return torch.tensor(tokenised, dtype=torch.long)