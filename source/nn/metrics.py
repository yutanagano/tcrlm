'Custom nn metric functions.'


import source.utils.metrics as metutils
import torch


@torch.no_grad()
def pretrain_accuracy(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor = None
) -> float:
    '''
    Calculate the accuracy of model predictions ignoring any padding tokens. If
    a mask is supplied, then only those residues that fall within the area
    where the mask evaluates to True will be considered.
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

    _, x_topk_indices = logits.topk(k, dim=-1, sorted=False)
    correct = (x_topk_indices == y)
    correct_masked = (correct & final_mask)

    return (correct_masked.sum() / total_residues_considered).item()


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

    cdr3_lens = metutils.get_cdr3_lens(x)
    start_indices, end_indices = metutils.get_cdr3_third(cdr3_lens, third)
    mask = metutils.get_cdr3_partial_mask(x, start_indices, end_indices)

    return pretrain_accuracy(logits, y, mask)


def pretrain_topk_accuracy_third(
    logits: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    third: int
) -> float:
    '''
    Calculate the top-5 accuracy of model predictions specifically of either
    the first, middle or final third segments of the CDR3s, where a prediction
    is considered correct if the correct option is within the top 5 predictions
    of the model.
    '''

    cdr3_lens = metutils.get_cdr3_lens(x)
    start_indices, end_indices = metutils.get_cdr3_third(cdr3_lens, third)
    mask = metutils.get_cdr3_partial_mask(x, start_indices, end_indices)

    return pretrain_topk_accuracy(logits, y, k, mask)


@torch.no_grad()
def finetune_accuracy(x: torch.Tensor, y: torch.Tensor) -> float:
    '''
    Calculate the accuracy of model predictions.
    '''

    correct = torch.argmax(x,dim=1) == y
    return (correct.sum() / correct.size(0)).item()