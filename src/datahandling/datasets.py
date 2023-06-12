"""
Custom dataset classes.
"""


import pandas as pd
from pathlib import Path

from .tokenisers import _Tokeniser
from torch.utils.data import Dataset
from typing import Union


class TCRDataset(Dataset):
    """
    Base dataset class to load and tokenise TCR data.
    """

    def __init__(
        self, data: Union[Path, str, pd.DataFrame], tokeniser: _Tokeniser
    ):
        """
        :param data: TCR data source
        :type data: str or Path (to csv) or DataFrame
        :param tokeniser: TCR tokeniser
        :type tokeniser: Tokeniser
        """
        super(TCRDataset, self).__init__()

        if type(data) != pd.DataFrame:
            data = pd.read_csv(
                data,
                dtype={
                    "TRAV": "string",
                    "CDR3A": "string",
                    "TRAJ": "string",
                    "TRBV": "string",
                    "CDR3B": "string",
                    "TRBJ": "string",
                    "Epitope": "string",
                    "MHCA": "string",
                    "MHCB": "string",
                    "duplicate_count": "UInt32",
                },
            )

        self._data = data
        self._tokeniser = tokeniser

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> any:
        return self._tokeniser.tokenise(self._data.iloc[index])


class AutoContrastiveDataset(TCRDataset):
    """
    Dataset for producing unsupervised contrastive loss pairs (x = x_prime).
    """

    def __init__(
        self,
        data: Union[Path, str, pd.DataFrame],
        tokeniser: _Tokeniser,
        censoring_lhs: bool,
        censoring_rhs: bool,
    ):
        super().__init__(data, tokeniser)
        self.censoring_lhs = censoring_lhs
        self.censoring_rhs = censoring_rhs

    def __getitem__(self, index: int) -> any:
        x = self._tokeniser.tokenise(self._data.iloc[index])
        x_lhs = self._tokeniser.tokenise(
            self._data.iloc[index], noising=self.censoring_lhs
        )
        x_rhs = self._tokeniser.tokenise(
            self._data.iloc[index], noising=self.censoring_rhs
        )

        return (x, x_lhs, x_rhs)


class EpitopeContrastiveDataset_dep(AutoContrastiveDataset):
    """
    Dataset for fetching epitope-matched TCR pairs from labelled data.

    In order to ensure equal balancing of all epitope groups, each "sample"
    from this dataset actually consists of one sample (TCR pair) from each
    epitope group.

    This means that the size of the dataset is no longer the number of rows
    in the underlying dataframe, but actually the number of rows which is
    represented by the largest epitope group in the dataset.
    """

    def __init__(
        self,
        data: Union[Path, str, pd.DataFrame],
        tokeniser: _Tokeniser,
        censoring_lhs: bool,
        censoring_rhs: bool,
    ):
        super().__init__(data, tokeniser, censoring_lhs, censoring_rhs)

        self._eps = self._data["Epitope"].unique()
        self._ep_groupby = self._data.groupby("Epitope")
        self._len = self._ep_groupby.size().max()

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> any:
        return [self._generate_matched_pair(index, ep) for ep in self._eps]

    def _generate_matched_pair(self, index: int, epitope: str) -> any:
        # Translate index to within epitope group
        ep_idx = index % self._ep_groupby.size()[epitope]

        # Sample pair
        x_row = self._ep_groupby.get_group(epitope).iloc[ep_idx]
        x_prime_row = self._ep_groupby.get_group(epitope).sample().iloc[0]

        # Tokenise pair
        x = self._tokeniser.tokenise(x_row)
        x_lhs = self._tokeniser.tokenise(x_row, noising=self.censoring_lhs)
        x_rhs = self._tokeniser.tokenise(x_prime_row, noising=self.censoring_rhs)

        return (x, x_lhs, x_rhs)

    def _internal_shuffle(self, random_seed: int) -> None:
        """
        Shuffles the dataframe and regenerates the groupby object.
        This IN ADDITION to random sampling by the pytorch sampler is necessary
        to ensure that each epoch of the dataset can generate fully unique ways
        of combining different TCR pairs in every sample. This is because every
        "sample" has samples from every epitope group and this mapping is
        partially deterministic given an index.

        This method should be called together with the set_epoch method of the
        pytorch distributed sampler.
        """
        self._data = self._data.sample(frac=1, random_state=random_seed)
        self._ep_groupby = self._data.groupby("Epitope")


class EpitopeContrastiveDataset(AutoContrastiveDataset):
    def __init__(self, data: Path | str | pd.DataFrame, tokeniser: _Tokeniser, censoring_lhs: bool, censoring_rhs: bool):
        super().__init__(data, tokeniser, censoring_lhs, censoring_rhs)

        self._ep_tcr_counts = self._data["Epitope"].value_counts()
        self._ep_pair_counts = {
            epitope: tcr_count * (tcr_count - 1) // 2
            for epitope, tcr_count in self._ep_tcr_counts.items()
        }
        self._ep_groupby = self._data.groupby("Epitope")
        self._len = sum(
            [pair_count for _, pair_count in self._ep_pair_counts.items()]
        )
    
    def __len__(self) -> int:
        return self._len
    
    def __getitem__(self, index: int) -> any:
        x_row, x_prime_row = self._get_pair(index)

        # Tokenise pair
        x = self._tokeniser.tokenise(x_row)
        x_lhs = self._tokeniser.tokenise(x_row, noising=self.censoring_lhs)
        x_rhs = self._tokeniser.tokenise(x_prime_row, noising=self.censoring_rhs)

        return (x, x_lhs, x_rhs)
    
    def _get_pair(self, index: int) -> tuple:
        epitope, internal_pair_idx = self._get_internal_pair_idx(index)
        x_internal_idx, x_prime_internal_idx = self._get_internal_idcs(epitope, internal_pair_idx)

        x_row = self._ep_groupby.get_group(epitope).iloc[x_internal_idx]
        x_prime_row = self._ep_groupby.get_group(epitope).iloc[x_prime_internal_idx]

        return x_row, x_prime_row
    
    def _get_internal_pair_idx(self, index: int) -> tuple:
        """
        Given an external index value, and given the combinatorial group sizes
        of all epitope groups, compute from which epitope group the indexed
        pair should come from, and compute the within-epitope-group index of
        the specified pair.
        """
        if index >= 0:
            internal_pair_idx = index
        # Handle negative indices
        else:
            internal_pair_idx = self._len + index
            # Handle negative index out of range
            if internal_pair_idx < 0:
                raise IndexError(f"Index {index} out of range.")

        for epitope, pair_count in self._ep_pair_counts.items():
            # If the index value falls within the pair count, it must be from
            # that epitope group, and the value in internal_pair_idx must be
            # the within-epitope-group index as well.
            if internal_pair_idx < pair_count:
                return epitope, internal_pair_idx
            
            # The internal_pair_idx value is larger than the pair count of the
            # epitope group currently in consideration, so it must be in some
            # later epitope group. Subtract the pair count of the current
            # epitope group so that the reference point of the index is from
            # the zeroth index of the following epitope group, then repeat.
            internal_pair_idx -= pair_count
        
        raise IndexError(f"Index {index} out of range, or something went terribly wrong!")
    
    def _get_internal_idcs(self, epitope: str, internal_pair_idx: int) -> tuple:
        """
        Given an epitope group and an internal pair index, uses logic similar
        to _get_internal_pair_idx to get the within-epitope-group TCR row index
        of the left hand side (lhs) and right hand side (rhs) of the positive
        TCR pair.
        """
        tcr_count = self._ep_tcr_counts[epitope]
        rhs_idx = internal_pair_idx

        for lhs_idx in range(tcr_count-1):
            current_lhs_pair_count = tcr_count-1-lhs_idx

            if rhs_idx < current_lhs_pair_count:
                # Adjust rhs_idx because the rhs_idx currently counts how many
                # rows down from the row below lhs the rhs is
                return lhs_idx, (lhs_idx + 1 + rhs_idx)
            
            rhs_idx -= current_lhs_pair_count
        
        raise IndexError(f"Internal pair index {internal_pair_idx} for epitope {epitope} out of range!")