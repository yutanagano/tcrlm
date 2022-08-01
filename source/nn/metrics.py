'Custom nn metric functions.'

import torch
from typing import Tuple


# Training metric functions
@torch.no_grad()
def pretrain_accuracy(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor = None
) -> float:
    '''
    Calculate the accuracy of model predictions ignoring any padding tokens. If
    a mask is supplied, then only those residues that fall within the area where
    the mask evaluates to True will be considered.
    '''
    final_mask = (y != 21) # ignore any padding tokens
    if mask is not None:
        # combine with supplied mask if exists
        final_mask = final_mask & mask

    # If no residues can be considered for the current batch, return None
    total_residues_considered = final_mask.sum()
    if total_residues_considered.item() == 0:
        return None
    
    correct = (torch.argmax(logits,dim=-1) == y)
    correct_masked = (correct & final_mask)

    return (correct_masked.sum() / total_residues_considered).item()


@torch.no_grad()
def pretrain_topk_accuracy(
    logits: torch.Tensor,
    y: torch.Tensor,
    k: int,
    mask: torch.Tensor = None
) -> float:
    '''
    Calculate the top-5 accuracy of model predictions ignoring any padding
    tokens, where a prediction is considered correct if the correct option is
    within the top 5 predictions of the model. If a mask is supplied, then only
    those residues that fall within the area where the mask evaluates to True
    will be considered.
    '''
    final_mask = (y != 21) # ignore any padding tokens
    if mask is not None:
        # combine with supplied mask if exists
        final_mask = final_mask & mask

    # If no residues can be considered for teh current batch, return None
    total_residues_considered = final_mask.sum()
    if total_residues_considered.item() == 0:
        return None

    y = y.unsqueeze(-1)
    final_mask = final_mask.unsqueeze(-1)

    x_topk_values, x_topk_indices = logits.topk(k, dim=-1, sorted=False)
    correct = (x_topk_indices == y)
    correct_masked = (correct & final_mask)

    return (correct_masked.sum() / total_residues_considered).item()


@torch.no_grad()
def _get_cdr3_lens(x: torch.Tensor) -> torch.Tensor:
    '''
    Given a 2D tensor representing a batch of tokenised CDR3s with padding, get
    the lengths of each CDR3 collected as a 1D tensor.
    '''
    cdr3_mask = (x != 21)
    return torch.count_nonzero(cdr3_mask, dim=-1)


@torch.no_grad()
def _get_cdr3_third(
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
def _get_cdr3_partial_mask(
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


def pretrain_accuracy_third(
    logits: torch.Tensor,
    x: torch.tensor,
    y: torch.Tensor,
    third: int
) -> float:
    '''
    Calculate the accuracy of model predictions specifically looking at either
    the first, middle or final third segments of the CDR3s.
    '''
    cdr3_lens = _get_cdr3_lens(x)
    start_indices, end_indices = _get_cdr3_third(cdr3_lens, third)
    mask = _get_cdr3_partial_mask(x, start_indices, end_indices)

    return pretrain_accuracy(logits, y, mask)


def pretrain_topk_accuracy_third(
    logits: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    third: int
) -> float:
    '''
    Calculate the top-5 accuracy of model predictions specifically of either the
    first, middle or final third segments of the CDR3s, where a prediction is
    considered correct if the correct option is within the top 5 predictions of
    the model.
    '''
    cdr3_lens = _get_cdr3_lens(x)
    start_indices, end_indices = _get_cdr3_third(cdr3_lens, third)
    mask = _get_cdr3_partial_mask(x, start_indices, end_indices)

    return pretrain_topk_accuracy(logits, y, k, mask)


@torch.no_grad()
def finetune_accuracy(x: torch.Tensor, y: torch.Tensor) -> float:
    '''
    Calculate the accuracy of model predictions.
    '''
    correct = torch.argmax(x,dim=1) == y
    return (correct.sum() / correct.size(0)).item()