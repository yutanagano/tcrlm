"""
Custom dataset classes.
"""


import pandas as pd
from pathlib import Path
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

    def __getitem__(self, index: int) -> any:
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

    def __getitem__(self, index: int) -> any:
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
        tokeniser: tokenisers._Tokeniser,
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
