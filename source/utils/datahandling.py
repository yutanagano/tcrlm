'Utility/helper resources involving data (pre)processing.'


import pandas as pd
import torch
from typing import Union


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


def check_dataframe_format(dataframe: pd.DataFrame, columns: list) -> None:
    if dataframe.columns.tolist() != columns:
        raise RuntimeError(
            f'CSV file with incompatible format: columns '
            f'{dataframe.columns.tolist()}, expected {columns}.'
        )