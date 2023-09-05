import torch
from torch import FloatTensor, LongTensor, Tensor
from typing import Optional


@torch.no_grad()
def mlm_acc(logits: FloatTensor, y: LongTensor, mask: Optional[Tensor] = None) -> float:
    """
    Calculate the accuracy of model mlm predictions ignoring any padding
    tokens. If a mask is supplied, then only those residues that fall within
    the area where the mask evaluates to True will be considered.
    """
    final_mask = y != 0  # ignore any padding tokens
    if mask is not None:
        # combine with supplied mask if exists
        final_mask = final_mask & mask

    # If no residues can be considered for the current batch, return None
    total_residues_considered = final_mask.sum()
    if total_residues_considered.item() == 0:
        return None

    correct = torch.argmax(logits, dim=-1) == (
        y - 3
    )  # minus two to translate from input vocabulary space to output vocabulary space
    correct_masked = correct & final_mask

    return (correct_masked.sum() / total_residues_considered).item()


@torch.no_grad()
def mlm_topk_acc(logits: FloatTensor, y: LongTensor, k: int, mask: Tensor = None) -> float:
    """
    Calculate the top-5 accuracy of model mlm predictions ignoring any padding
    tokens, where a prediction is considered correct if the correct option is
    within the top k predictions of the model. If a mask is supplied, then only
    those residues that fall within the area where the mask evaluates to True
    will be considered.
    """
    final_mask = y != 0  # ignore any padding tokens
    if mask is not None:
        # combine with supplied mask if exists
        final_mask = final_mask & mask

    # If no residues can be considered for teh current batch, return None
    total_residues_considered = final_mask.sum()
    if total_residues_considered.item() == 0:
        return None

    y = y.unsqueeze(-1)
    final_mask = final_mask.unsqueeze(-1)

    _, x_topk_indices = logits.topk(k, dim=-1, sorted=False)
    correct = x_topk_indices == (
        y - 3
    )  # minus two to translate from input vocabulary space to output vocabulary space
    correct_masked = correct & final_mask

    return (correct_masked.sum() / total_residues_considered).item()