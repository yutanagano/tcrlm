from abc import ABC, abstractmethod
import torch
from torch import BoolTensor, FloatTensor
from torch.nn import Module


class BatchContrastiveLoss(ABC, Module):
    """
            1    N
    Loss = --- * Σ (Loss_i)
            N    i

                     -1                           exp(-d(z_i, z_p) / t)
    where Loss_i = ------  *    Σ      log ( ------------------------------ )
                   |P(i)|    p in P(i)          Σ     exp(-d(z_i, z_a) / t)
                                             a in A(i)

    where
        i is the index of an in-batch sample
        d is some distance function defined over the product of the sample space
        P(i) is the set of positives of element i
        A(i) is the set of all elements other than i
        t is the temperature hyperparameter

    and Loss_i is euivalent to (and here is computed in this way*):

                                                        1
    Loss_i = log (    Σ     exp(-d(z_i, z_a) / t) ) + ------    Σ      (d(z_i, z_p) / t)
                   a in A(i)                          |P(i)| p in P(i)
             ______________________________________   __________________________________
                        background term                         positives term

    *A caveat is that in order to prevent the exp terms from suffering from overflow, the exponent terms (d/t) are all normalised.
    This is done by subtracting the maximum value of each row in the exponent score matrix from all values in the same row.
    However, due to the properties of softmax-like functions, this translation does not affect the computed loss value.
    This procedure only serves to ensure that the computation behaves even with very low temperatures.
    """

    @staticmethod
    @abstractmethod
    def pdist_squareform(tcr_representations: FloatTensor) -> FloatTensor:
        """
        Placeholder for arbitrary distance function.
        Accepts representations and returns a squareform pdist matrix.
        """
        pass

    def __init__(self, temp: float = 0.05) -> None:
        super().__init__()
        self._temp = temp

    def forward(
        self, tcr_representations: FloatTensor, positives_mask: BoolTensor
    ) -> FloatTensor:
        exponent_terms = -self.pdist_squareform(tcr_representations) / self._temp
        exponent_terms = self._adjust_to_prevent_overflow(exponent_terms)
        positives_term = self._get_positives_term(exponent_terms, positives_mask)
        background_term = self._get_background_term(exponent_terms)
        loss_per_sample = background_term - positives_term

        return loss_per_sample.mean()

    @staticmethod
    def _adjust_to_prevent_overflow(exponent_terms: FloatTensor) -> FloatTensor:
        ALONG_ROWS = 1
        value_guaranteed_to_be_small = exponent_terms.min().item()
        exponent_terms.fill_diagonal_(value_guaranteed_to_be_small)
        max_off_diagonal_value_from_each_row, _ = exponent_terms.max(dim=ALONG_ROWS, keepdim=True)
        return exponent_terms - max_off_diagonal_value_from_each_row

    def _get_positives_term(
        self, exponent_terms: FloatTensor, positives_mask: BoolTensor
    ) -> FloatTensor:
        ALONG_COMPARISONS_DIM = 1
        contributions_from_positive_comparisons = torch.sum(
            exponent_terms * positives_mask, dim=ALONG_COMPARISONS_DIM
        )
        num_positives_per_sample = positives_mask.sum(dim=ALONG_COMPARISONS_DIM)
        return contributions_from_positive_comparisons / num_positives_per_sample

    def _get_background_term(self, exponent_terms: FloatTensor) -> FloatTensor:
        ALONG_COMPARISONS_DIM = 1
        identity_mask = torch.eye(len(exponent_terms), device=exponent_terms.device)
        non_identity_mask = torch.ones_like(identity_mask) - identity_mask
        exp_dot_products = torch.exp(exponent_terms)
        contributions_from_non_identity_terms = torch.sum(
            exp_dot_products * non_identity_mask, dim=ALONG_COMPARISONS_DIM
        )
        return torch.log(contributions_from_non_identity_terms)


class DotProductLoss(BatchContrastiveLoss):
    @staticmethod
    def pdist_squareform(tcr_representations: FloatTensor) -> FloatTensor:
        return -torch.matmul(tcr_representations, tcr_representations.T)
    

class EuclideanDistanceLoss(BatchContrastiveLoss):
    @staticmethod
    def pdist_squareform(tcr_representations: FloatTensor) -> FloatTensor:
        return torch.cdist(tcr_representations, tcr_representations, p=2)


class AngularDistanceLoss(BatchContrastiveLoss):
    @staticmethod
    def pdist_squareform(tcr_representations: FloatTensor) -> FloatTensor:
        cosine_similarity = torch.matmul(tcr_representations, tcr_representations.T) # note that all representations are l2-normed beforehand
        cosine_similarity = torch.clamp(cosine_similarity, min=-1, max=1)
        return torch.acos(cosine_similarity)