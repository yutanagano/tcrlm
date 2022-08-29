'Miscellaneous utilities used for training.'


import os
import pandas as pd
import torch
from typing import Union
from statistics import fmean


def print_with_deviceid(msg: str, device: torch.device) -> None:
    print(f'[{device}]: {msg}')


def set_env_vars(master_addr: str, master_port: str) -> None:
    '''
    Set some environment variables that are required when spawning parallel
    processes using torch.nn.parallel.DistributedDataParallel.
    '''

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port


def dynamic_fmean(l: list):
    '''
    Dynamically calculate average of list containing training metric values.
    First, cleans the list of any invalid (None) values, then calculates
    average of remaining values. If no values remain, output string 'n/a'.
    '''

    l = [x for x in l if x is not None]
    if len(l) == 0: return 'n/a'
    return fmean(l)


def check_dataframe_format(dataframe: pd.DataFrame, columns: list) -> None:
    if dataframe.columns.tolist() != columns:
        raise RuntimeError(
            f'CSV file with incompatible format: columns '
            f'{dataframe.columns.tolist()}, expected {columns}.'
        )


# Tokenisation tools
amino_acids = {
    'A','C','D','E','F','G','H','I','K','L',
    'M','N','P','Q','R','S','T','V','W','Y'
}

tokens = (
    'A','C','D','E','F','G','H','I','K','L', # amino acids
    'M','N','P','Q','R','S','T','V','W','Y', # (0-19)
    '?', # mask token (20)
    '-'  # padding token (21)
)
token_to_index = dict()
index_to_token = dict()
for t, i in zip(tokens, range(len(tokens))):
    token_to_index[t] = i
    index_to_token[i] = t


def tokenise(cdr3: Union[list, str]) -> torch.Tensor:
    'Turn a cdr3 sequence from list form to tokenised tensor form.'

    cdr3 = map(lambda x: token_to_index[x], cdr3)
    return torch.tensor(list(cdr3), dtype=torch.long)