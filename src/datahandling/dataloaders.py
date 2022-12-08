'''
Custom dataloader classes.
'''


import random
from src.datahandling import datasets
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Union


class TCRDataLoader(DataLoader):
    '''
    Base dataloader class.
    '''
    def __init__(
        self,
        dataset: datasets.TCRDataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = True,
        num_workers: int = 0
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )


    def collate_fn(self, batch) -> Union[Tuple[Tensor], Tensor]:
        '''
        Pad and batch tokenised TCRs.
        '''
        elem = batch[0]

        if isinstance(elem, list) or isinstance(elem, tuple):
            return tuple(
                map(
                    lambda x: pad_sequence(
                        sequences=x,
                        batch_first=True,
                        padding_value=0
                    ),
                    zip(*batch)
                )
            )

        return pad_sequence(
            sequences=batch,
            batch_first=True,
            padding_value=0
        )


class MLMDataLoader(TCRDataLoader):
    '''
    Masked-language modelling dataloader class.
    '''
    def __init__(
        self,
        dataset: datasets.TCRDataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        p_mask: float = 0.15,
        p_mask_random: float = 0.1,
        p_mask_keep: float = 0.1
    ):
        if p_mask < 0 or p_mask >= 1:
            raise RuntimeError(f'p_mask must lie in [0,1): {p_mask}')
        
        if p_mask_random < 0 or p_mask_random > 1:
            raise RuntimeError(
                f'p_mask_random must lie in [0,1]: {p_mask_random}'
            )

        if p_mask_keep < 0 or p_mask_keep > 1:
            raise RuntimeError(
                f'p_mask_keep must lie in [0,1]: {p_mask_keep}'
            )

        if p_mask_random + p_mask_keep > 1:
            raise RuntimeError(
                'p_mask_random + p_mask_keep must be less than 1.'
            )

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            num_workers
        )

        self._vocabulary = set(range(3, dataset._tokeniser.vocab_size+3))
        self._p_mask = p_mask
        self._p_mask_random = p_mask_random
        self._p_mask_keep = p_mask_keep


    def _pick_masking_indices(self, seq_len: int) -> list:
        '''
        Decide on a set of token indices to mask. Never mask the first token,
        as it is reserved for the <cls> token, which will not be used during
        MLM.
        '''
        if self._p_mask == 0:
            return []
        
        num_to_be_masked = max(1, round(seq_len * self._p_mask))
        return random.sample(range(1, seq_len), num_to_be_masked)


    def _generate_masked(self, x: Tensor, indices: list) -> Tensor:
        x = x.detach().clone()

        for idx in indices:
            r = random.random()
            if r < self._p_mask_random:
                x[idx,0] = random.choice(tuple(self._vocabulary-{x[idx,0]}))
                continue
            
            if r < 1-self._p_mask_keep:
                x[idx,0] = 1
                continue

        return x


    def _generate_target(self, x: Tensor, indices: list) -> Tensor:
        target = torch.zeros_like(x[:,0])

        for idx in indices:
            target[idx] = x[idx,0]
        
        return target


    def _make_mlm_pair(self, x: Tensor) -> Tuple[Tensor]:
        seq_len = len(x)

        indices_to_mask = self._pick_masking_indices(seq_len)

        masked = self._generate_masked(x, indices_to_mask)
        target = self._generate_target(x, indices_to_mask)

        return (masked, target)


    def collate_fn(self, batch) -> Union[Tuple[Tensor], Tensor]:
        batch = [self._make_mlm_pair(x) for x in batch]

        return super().collate_fn(batch)


class UnsupervisedSimCLDataLoader(MLMDataLoader):
    '''
    Dataloader for unsupervised contrastive loss training.
    '''
    def collate_fn(self, batch) -> Union[Tuple[Tensor], Tensor]:
        batch = [(x, x_prime, *self._make_mlm_pair(x)) for x, x_prime in batch]

        return super(MLMDataLoader, self).collate_fn(batch)


class SupervisedSimCLDataLoader(UnsupervisedSimCLDataLoader):
    def __init__(
        self,
        dataset: datasets.SupervisedSimCLDataset,
        num_workers: int = 0,
        p_mask: float = 0.15,
        p_mask_random: float = 0.1,
        p_mask_keep: float = 0.1
    ):
        batch_size = dataset._num_epitopes
        super().__init__(
            dataset,
            batch_size,
            False,
            num_workers,
            p_mask,
            p_mask_random,
            p_mask_keep
        )