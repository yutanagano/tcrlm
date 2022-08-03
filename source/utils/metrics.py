'Helper functions for custom model metrics'


import torch
from typing import Tuple


@torch.no_grad()
def get_cdr3_lens(x: torch.Tensor) -> torch.Tensor:
    '''
    Given a 2D tensor representing a batch of tokenised CDR3s with padding, get
    the lengths of each CDR3 collected as a 1D tensor.
    '''

    cdr3_mask = (x != 21)
    return torch.count_nonzero(cdr3_mask, dim=-1)


@torch.no_grad()
def get_cdr3_third(
    lens: torch.Tensor,
    third: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Given the lengths of various CDR3s, calculate where the first, second or
    final thirds of the sequence would begin and end, and output the results
    as two 1D tensors. The parameter 'third' designates for which third the
    output should correspond. The first output tensor contains the starting
    indices of the designated third, and the second tensor contains the ending
    indices.
    '''

    first_third = (lens / 3).round().to(torch.long)
    second_third = (lens * 2 / 3).round().to(torch.long)

    if third == 0:
        return (torch.zeros_like(lens), first_third)
    elif third == 1:
        return (first_third, second_third)
    elif third == 2:
        return (second_third, lens)
    else:
        raise RuntimeError(
            'The parameter third takes an integer value between 0-2. '
            f'(value provided: {third})'
        )


@torch.no_grad()
def get_cdr3_partial_mask(
    x: torch.Tensor,
    start_indices: torch.Tensor,
    end_indices: torch.Tensor
) -> torch.Tensor:
    '''
    Given the y tensor and two tensors representing the starting and ending
    indices for the regions of interest for each CDR3, generate a mask
    highlighting only the region of interest for each CDR3.
    '''

    mask = torch.zeros(
        size=(x.size(0), x.size(1) + 1),
        dtype=torch.long,
        device=x.device
    )
    mask[(torch.arange(mask.size(0)),start_indices)] = 1
    mask[(torch.arange(mask.size(0)),end_indices)] += -1
    mask = mask.cumsum(dim=1)[:,:-1]

    return mask