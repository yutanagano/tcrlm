'''
training.py
purpose: Python module with helper classes for training CDR3Bert.
author: Yuta Nagano
ver: 1.0.0
'''


import torch


def create_padding_mask(x: torch.Tensor) -> torch.Tensor:
    return x == 21