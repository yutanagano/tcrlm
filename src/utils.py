'''
Utility classes and functions.
'''


import json
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from src.datahandling.dataloaders import TCRDataLoader
from src.datahandling.datasets import TCRDataset
from src.datahandling.tokenisers import _Tokeniser
from src.models.embedder import _Embedder
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from typing import Dict


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
    model.cpu()
    if isinstance(model, DistributedDataParallel):
        model = model.module
    torch.save(model.state_dict(), save_dir/'state_dict.pt')

    # Save log
    pd.DataFrame.from_dict(log, orient='index')\
        .to_csv(save_dir/'log.csv', index_label='epoch')

    # Save config
    with open(save_dir/'config.json', 'w') as f:
        json.dump(config, f, indent=4)


class PCDistModelWrapper:
    '''
    Pytorch embedder model wrapper to faciliate model sharing between different
    evaluation pipelines.
    '''
    def __init__(
        self,
        model: _Embedder,
        model_name: str,
        tokeniser: _Tokeniser
    ) -> None:
        self.module = model.eval()
        self._name = model_name
        self._tokeniser = tokeniser


    @property
    def name(self) -> str:
        return self._name

    
    def pdist(self, sequence_elements: Dict[str, ndarray]) -> ndarray:
        dl = self._generate_dataloader(sequence_elements)
        embedded = self._generate_embeddings(dl)

        return torch.pdist(embedded, p=2).detach().numpy()


    def cdist(
        self,
        sequence_elements_1: Dict[str, ndarray],
        sequence_elements_2: Dict[str, ndarray]
    ) -> ndarray:
        dl_1 = self._generate_dataloader(sequence_elements_1)
        dl_2 = self._generate_dataloader(sequence_elements_2)

        embedded_1 = self._generate_embeddings(dl_1)
        embedded_2 = self._generate_embeddings(dl_2)

        return torch.cdist(embedded_1, embedded_2, p=2).detach().numpy()


    def _generate_dataloader(
        self,
        sequence_elements: Dict[str, ndarray]
    ) -> TCRDataLoader:
        # Create missing columns if any and mark as empty
        for col in (
            'TRAV',
            'CDR3A',
            'TRAJ',
            'TRBV',
            'CDR3B',
            'TRBJ',
            'Epitope',
            'MHCA',
            'MHCB',
            'duplicate_count'
        ):
            if col not in sequence_elements:
                sequence_elements[col] = None
        
        # Generate dataframe, dataset, then wrap in dataloader
        df = DataFrame.from_dict(sequence_elements)
        ds = TCRDataset(data=df, tokeniser=self._tokeniser)
        return TCRDataLoader(dataset=ds, batch_size=512, shuffle=False)


    @torch.no_grad()
    def _generate_embeddings(self, dataloader: TCRDataLoader) -> Tensor:
        embedded = [self.module.embed(batch) for batch in dataloader]
        return torch.concat(embedded)