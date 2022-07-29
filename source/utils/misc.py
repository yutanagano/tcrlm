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


def compare_models(dirpath: str, n_gpus: int) -> None:
    '''
    Compare all models saved from distributed processes and ensure that they
    are all equivalent i.e. that they all have the same weights.

    Used for integration testing.
    '''

    print('Comparing model enpoints from distributed training...')

    paths_to_models = [
        os.path.join(dirpath, file) for file in os.listdir(dirpath) \
        if file.endswith('.ptnn')
    ]

    assert len(paths_to_models) == n_gpus

    print('Model endpoints detected:')
    for path in paths_to_models:
        print(path)

    models = [torch.load(path) for path in paths_to_models]

    def models_equivalent(model1: torch.nn.Module, model2: torch.nn.Module):
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if not torch.equal(p1, p2):
                return False
        return True
    
    for i, model2 in enumerate(models[1:], start=1):
        if not models_equivalent(models[0], model2):
            raise RuntimeError(
                f'Models {paths_to_models[0]} and {paths_to_models[i]} '
                'are not equivalent.'
            )
    
    print('All model endpoints compared, no discrepancies found!')