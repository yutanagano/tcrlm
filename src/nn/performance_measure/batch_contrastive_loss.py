import torch
from torch import BoolTensor, FloatTensor
from torch.nn import Module


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

    def forward(self, tcr_representations: FloatTensor, positives_mask: BoolTensor) -> FloatTensor:
        dot_products = torch.matmul(tcr_representations, tcr_representations.T) / self._temp
        positives_term = self._get_positives_term(dot_products, positives_mask)
        background_term = self._get_background_term(dot_products)
        loss_per_sample = background_term - positives_term
        return loss_per_sample.mean()

    def _get_positives_term(self, dot_products: FloatTensor, positives_mask: BoolTensor) -> FloatTensor:
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