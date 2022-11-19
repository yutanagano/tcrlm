'''
Utility classes and functions.
'''


import json
import pandas as pd
from pathlib import Path
import torch
from torch import Tensor
from torch.nn import Module


def masked_average_pool(
    x: Tensor,
    padding_mask: Tensor
) -> Tensor:
    '''
    Given a tensor x, representing a batch of length-padded token embedding
    sequences, and a corresponding padding mask tensor, compute the average
    pooled vector for every sequence in the batch with the padding taken into
    account.

    :param x: Tensor representing a batch of length-padded token embedding
        sequences, where indices of tokens are marked with 1s and indices of
        paddings are marked with 0s (size B,L,E)
    :type x: torch.Tensor
    :param padding_mask: Tensor representing the corresponding padding mask
        (size B,L)
    :type padding_mask: torch.Tensor
    :return: Average pool of token embeddings per sequence (size B,E)
    '''
    # Reverse the boolean values of the mask to mark where the tokens are, as
    # opposed to where the tokens are not. Then, resize padding mask to make it
    # broadcastable with token embeddings
    padding_mask = padding_mask.logical_not().unsqueeze(-1)

    # Compute averages of token embeddings per sequence, ignoring padding tokens
    token_embeddings_masked = x * padding_mask
    token_embeddings_summed = token_embeddings_masked.sum(1)
    token_embeddings_averaged = token_embeddings_summed / padding_mask.sum(1)

    return token_embeddings_averaged


def save(
    wd: Path,
    save_name: str,
    model: Module,
    log: dict,
    config: dict
) -> None:
    model_saves_dir = wd/'model_saves'
    try:
        model_saves_dir.mkdir()
    except(FileExistsError):
        pass

    try:
        (model_saves_dir/save_name).mkdir()
    except(FileExistsError):
        suffix_int = 1
        new_save_name = f'{save_name}_{suffix_int}'
        done = False
        while not done:
            try:
                (model_saves_dir/new_save_name).mkdir()
                save_name = new_save_name
                done = True
            except(FileExistsError):
                suffix_int += 1
                new_save_name = f'{save_name}_{suffix_int}'
    save_dir = model_saves_dir/save_name

    # Save model
    torch.save(model.state_dict(), save_dir/'state_dict.pt')

    # Save log
    pd.DataFrame.from_dict(log, orient='index')\
        .to_csv(save_dir/'log.csv', index_label='epoch')

    # Save config
    with open(save_dir/'config.json', 'w') as f:
        json.dump(config, f, indent=4)