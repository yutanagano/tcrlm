"""
Custom dataset classes.
"""


import pandas as pd
from pathlib import Path

from src.datahandling import tokenisers
from . import tokenisers
from torch.utils.data import Dataset
from typing import Union


class TCRDataset(Dataset):
    """
    Base dataset class to load and tokenise TCR data.
    """

    def __init__(
        self, data: Union[Path, str, pd.DataFrame], tokeniser: tokenisers._Tokeniser
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

    def __getitem__(self, index) -> any:
        return self._tokeniser.tokenise(self._data.iloc[index])


class AutoContrastiveDataset(TCRDataset):
    """
    Dataset for producing unsupervised contrastive loss pairs (x = x_prime).
    """

    def __init__(
        self,
        data: Union[Path, str, pd.DataFrame],
        tokeniser: tokenisers._Tokeniser,
        censoring_lhs: bool,
        censoring_rhs: bool,
    ):
        super().__init__(data, tokeniser)
        self.censoring_lhs = censoring_lhs
        self.censoring_rhs = censoring_rhs

    def __getitem__(self, index) -> any:
        x = self._tokeniser.tokenise(self._data.iloc[index])
        x_lhs = self._tokeniser.tokenise(
            self._data.iloc[index], noising=self.censoring_lhs
        )
        x_rhs = self._tokeniser.tokenise(
            self._data.iloc[index], noising=self.censoring_rhs
        )

        return (x, x_lhs, x_rhs)


class EpitopeContrastiveDataset(AutoContrastiveDataset):
    """
    Dataset for fetching epitope-matched TCR pairs from labelled data.
    """

    def __init__(
        self,
        data: Union[Path, str, pd.DataFrame],
        tokeniser: tokenisers._Tokeniser,
        censoring_lhs: bool,
        censoring_rhs: bool
    ):
        super().__init__(data, tokeniser, censoring_lhs, censoring_rhs)

        self._ep_groupby = self._data.groupby("Epitope")

    def __getitem__(self, index) -> any:
        x_row = self._data.iloc[index]
        x_prime_row = self._ep_groupby.get_group(x_row["Epitope"]).sample().iloc[0]

        x = self._tokeniser.tokenise(x_row)
        x_lhs = self._tokeniser.tokenise(x_row, noising=self.censoring_lhs)
        x_rhs = self._tokeniser.tokenise(x_prime_row, noising=self.censoring_rhs)

        return (x, x_lhs, x_rhs)
