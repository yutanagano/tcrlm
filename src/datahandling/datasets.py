'Custom dataset classes.'


import pandas as pd
from pathlib import Path
from src.datahandling import tokenisers
from torch.utils.data import Dataset
from typing import Union


class TcrDataset(Dataset):
    '''
    Base dataset class to load and tokenise TCR data.
    '''

    def __init__(
        self,
        data: Union[Path, str, pd.DataFrame],
        tokeniser: tokenisers.Tokeniser
    ):
        '''
        :param data: TCR data source
        :type data: str or Path (to csv) or DataFrame
        :param tokeniser: TCR tokeniser
        :type tokeniser: Tokeniser
        '''
        super(TcrDataset, self).__init__()

        if type(data) != pd.DataFrame:
            data = pd.read_csv(data, dtype='string')

        self._data = data
        self._tokeniser = tokeniser


    def __len__(self) -> int:
        return len(self._data)


    def __getitem__(self, index) -> any:
        return self._tokeniser.tokenise(self._data.iloc[index])