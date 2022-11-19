import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F


def alignment(x: Tensor, labels: Tensor, alpha: int = 2):
    '''
    Computes alignment between embeddings of instances belonging to the same
    class label, as specified by the labels tensor.
    '''

    num_cls = len(labels.unique())

    cls_views = [
        x[(labels == cls_code).nonzero().squeeze(dim=-1)]
        for cls_code in range(num_cls)
    ]

    cls_pdists = torch.concat(
        [torch.pdist(cls_view, p=2).pow(alpha) for cls_view in cls_views]
    )

    return cls_pdists.mean()


def uniformity(x: Tensor, t: int = 2):
    '''
    Computes an empirical estimate of uniformity given background data x.
    '''

    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()


@torch.no_grad()
def mlm_acc(logits: Tensor, y: Tensor, mask: Tensor = None) -> float:
    '''
    Calculate the accuracy of model mlm predictions ignoring any padding
    tokens. If a mask is supplied, then only those residues that fall within
    the area where the mask evaluates to True will be considered.
    '''

    final_mask = (y != 0) # ignore any padding tokens
    if mask is not None:
        # combine with supplied mask if exists
        final_mask = final_mask & mask

    # If no residues can be considered for the current batch, return None
    total_residues_considered = final_mask.sum()
    if total_residues_considered.item() == 0:
        return None
    
    correct = (torch.argmax(logits,dim=-1) == (y - 3)) # minus two to translate from input vocabulary space to output vocabulary space
    correct_masked = (correct & final_mask)

    return (correct_masked.sum() / total_residues_considered).item()


@torch.no_grad()
def mlm_topk_acc(
    logits: Tensor,
    y: Tensor,
    k: int,
    mask: Tensor = None
) -> float:
    '''
    Calculate the top-5 accuracy of model mlm predictions ignoring any padding
    tokens, where a prediction is considered correct if the correct option is
    within the top k predictions of the model. If a mask is supplied, then only
    those residues that fall within the area where the mask evaluates to True
    will be considered.
    '''

    final_mask = (y != 0) # ignore any padding tokens
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
    correct = (x_topk_indices == (y - 3)) # minus two to translate from input vocabulary space to output vocabulary space
    correct_masked = (correct & final_mask)

    return (correct_masked.sum() / total_residues_considered).item()


class AdjustedCELoss(CrossEntropyLoss):
    '''
    Custom cross entropy loss class which subtracts 2 from the input labels
    before running the cross entropy funciton. This is because our models'
    vocabulary space is indexed from 0...X whereas the target data's vocabulary
    is indexed from 2...X+2 (index 0 and 1 are reserved for the padding and
    masked tokens respectively).
    '''

    def __init__( self, label_smoothing: float = 0 ) -> None:
        super().__init__(label_smoothing=label_smoothing, ignore_index=-3)
        # ignore_index is set to -2 here because the padding token, indexed at
        # 0 in the target vocabulary space, will be mapped to -2 when
        # moving all indices two places down.

    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            input,
            target-3,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )