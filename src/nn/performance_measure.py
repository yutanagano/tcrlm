import torch
from torch import Tensor, FloatTensor, LongTensor
from torch.nn import CrossEntropyLoss, Module
from torch.nn import functional as F
from typing import Optional


class AdjustedCELoss(CrossEntropyLoss):
    """
    Custom cross entropy loss class which subtracts 2 from the input labels
    before running the cross entropy funciton. This is because our models'
    vocabulary space is indexed from 0...X whereas the target data's vocabulary
    is indexed from 2...X+2 (index 0 and 1 are reserved for the padding and
    masked tokens respectively).
    """

    def __init__(self, label_smoothing: float = 0) -> None:
        super().__init__(label_smoothing=label_smoothing, ignore_index=-3)
        # ignore_index is set to -2 here because the padding token, indexed at
        # 0 in the target vocabulary space, will be mapped to -2 when
        # moving all indices two places down.

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            input,
            target - 3,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class BatchContrastiveLoss(Module):
    """
            1    N
    Loss = --- * Σ (Loss_i)
            N    i
    
                     -1                            exp(z_i dot z_p / t)
    where Loss_i = ------  *    Σ      log ( ------------------------------ )
                   |P(i)|    p in P(i)          Σ      exp(z_i dot z_a / t)
                                             a in A(i)
    
    where
        i is the index of an in-batch sample
        P(i) is the set of positives of element i
        A(i) is the set of all elements other than i
        t is the temperature hyperparameter

    and Loss_i is euivalent to (and here is computed in this way):

                                                        1
    Loss_i = log (    Σ      exp(z_i dot z_a / t) ) - ------    Σ      (z_i dot z_p / t)
                   a in A(i)                          |P(i)| p in P(i)
             ______________________________________   __________________________________
                        background term                         positives term
    """

    def __init__(self, temp: float = 0.05) -> None:
        super().__init__()
        self._temp = temp

    def forward(self, tcr_representations: FloatTensor, positives_mask: LongTensor) -> FloatTensor:
        dot_products = torch.matmul(tcr_representations, tcr_representations.T) / self._temp
        positives_term = self._get_positives_term(dot_products, positives_mask)
        background_term = self._get_background_term(dot_products)
        loss_per_sample = background_term - positives_term
        return loss_per_sample.mean()

    def _get_positives_term(self, dot_products: FloatTensor, positives_mask: LongTensor) -> FloatTensor:
        ALONG_COMPARISONS_DIM = 1
        contributions_from_positive_comparisons = torch.sum(dot_products * positives_mask, dim=ALONG_COMPARISONS_DIM)
        num_positives_per_sample = positives_mask.sum(dim=ALONG_COMPARISONS_DIM)
        return contributions_from_positive_comparisons / num_positives_per_sample

    def _get_background_term(self, dot_products: FloatTensor) -> FloatTensor:
        ALONG_COMPARISONS_DIM = 1
        identity_mask = torch.eye(len(dot_products), device=dot_products.device)
        non_identity_mask = torch.ones_like(identity_mask) - identity_mask
        exp_dot_products = torch.exp(dot_products)
        contributions_from_non_identity_terms = torch.sum(exp_dot_products * non_identity_mask, dim=ALONG_COMPARISONS_DIM)
        return torch.log(contributions_from_non_identity_terms)


@torch.no_grad()
def mlm_acc(logits: Tensor, y: Tensor, mask: Optional[Tensor] = None) -> float:
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
def mlm_topk_acc(logits: Tensor, y: Tensor, k: int, mask: Tensor = None) -> float:
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