'Miscellaneous utilities used for training.'


import json
import os
import pandas as pd
from src.datahandling import tokenisers
from statistics import fmean
import torch


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


def instantiate_tokeniser(hyperparameters: dict) -> tokenisers.AaTokeniser:
    tokeniser_class_str = hyperparameters['tokeniser_class']
    tokeniser_hyperparams = json.loads(
        hyperparameters['tokeniser_hyperparams']
    )

    if tokeniser_class_str == 'AaTokeniser':
        len_tupet = tokeniser_hyperparams['len_tuplet']

        return tokenisers.AaTokeniser(len_tuplet=len_tupet)
    
    raise RuntimeError(f'Unrecognised tokeniser class: {tokeniser_class_str}')