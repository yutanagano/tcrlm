'Miscellaneous utilities used for training.'


import os
import torch
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