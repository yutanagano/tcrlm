import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.nn import functional as F
from typing import Optional


def alignment(z: Tensor, labels: Tensor, alpha: int = 1) -> Tensor:
    """
    Computes alignment between embeddings of instances belonging to the same
    class label, as specified by the labels tensor. It is assumed that the
    probability of getting TCRs against any epitope is equal.
    """
    num_cls = len(labels.unique())

    cls_views = [
        z[(labels == cls_code).nonzero().squeeze(dim=-1)] for cls_code in range(num_cls)
    ]

    cls_pdists = torch.stack(
        [torch.pdist(cls_view, p=2).pow(alpha).mean() for cls_view in cls_views]
    )

    return cls_pdists.mean()


def alignment_paired(z: Tensor, z_prime: Tensor, alpha: int = 1) -> Tensor:
    """
    Computes alignment between pairs of known positive-pair embeddings.
    """
    return (z - z_prime).norm(dim=1).pow(alpha).mean()


def uniformity(z: Tensor, alpha: int = 1, t: float = 1) -> Tensor:
    """
    Computes an empirical estimate of uniformity given background data x.
    """
    sq_pdist = torch.pdist(z, p=2).pow(alpha)
    return sq_pdist.mul(-t).exp().mean().log()


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


class SimCLoss(Module):
    """
    Simple contrastive loss based on SimCSE.
    """

    def __init__(self, temp: float = 0.05) -> None:
        super().__init__()
        self._temp = temp

    def forward(self, z: Tensor, z_prime: Tensor) -> Tensor:
        """
        Implements the simple contrastive function as seen in SimCSE. Assumes
        that embeddings (z, z_prime) are all already l2-normalised.
        """
        # z.size: (N,E), z_prime.T.size: (E,N)
        z_sim = torch.exp(torch.matmul(z, z_prime.T) / self._temp)  # (N,N)
        pos_sim = torch.diag(z_sim)  # (N,)
        back_sim = torch.sum(z_sim, dim=1)  # (N,)
        closs = -torch.log(pos_sim / back_sim)

        return closs.mean()


class SimECLoss(Module):
    """
    Simple Euclidean contrastive loss.
    """

    def __init__(self, temp: float=0.05) -> None:
        super().__init__()
        self._temp = temp

    def forward(self, z: Tensor, z_prime: Tensor) -> Tensor:
        z_sim = torch.exp(-torch.cdist(z, z_prime, p=2) / self._temp)  # (N,N)
        pos_sim = torch.diag(z_sim)  # (N,)
        back_sim = torch.sum(z_sim, dim=1)  # (N,)
        closs = -torch.log(pos_sim / back_sim)

        return closs.mean()


class SimCLoss2(Module):
    """
    A version of SimCLoss where the uniformity is calculated over the lhs.
    """

    def __init__(self, temp: float = 0.05) -> None:
        super().__init__()
        self._temp = temp

    def forward(self, z: Tensor, z_prime: Tensor) -> Tensor:
        # z.size: (N,E), z_prime.size: (N,E)
        pos_sim = torch.exp(torch.sum(z * z_prime, dim=-1) / self._temp)  # (N,)

        z_sim = torch.exp(torch.matmul(z, z.T) / self._temp)  # (N,N)
        z_sim = z_sim * (1 - torch.eye(z.size(0))).to(z_sim.device)  # (N,N), zero diag
        back_sim = torch.sum(z_sim, dim=-1)  # (N,)

        closs = -torch.log(pos_sim / back_sim)

        return closs.mean()


class AULoss(Module):
    """
    A loss calculated as alignment + uniformity over a matched-pair batch.
    """

    def __init__(self, alpha: int = 1, t: float = 1) -> None:
        super().__init__()
        self._alpha = alpha
        self._t = t

    def forward(self, z: Tensor, z_prime: Tensor) -> Tensor:
        return alignment_paired(z, z_prime, alpha=self._alpha) + 0.5 * (
            uniformity(z, alpha=self._alpha, t=self._t)
            + uniformity(z_prime, alpha=self._alpha, t=self._t)
        )


class PosBackSimCLoss(Module):
    """
    Simple contrastive loss, but where the positive pairs can be sampled
    independently of the background pairs.
    """

    def __init__(self, temp: float = 0.05) -> None:
        super().__init__()
        self._temp = temp

    def forward(self, z: Tensor, z_pos: Tensor, z_back: Tensor) -> Tensor:
        """
        NOTE: assumes that embeddings are all already l2-normalised.
        """
        # z.size: (N,E), z_pos.size: (N,E), z_back.T.size: (E,M)
        pos_sim = torch.exp((z * z_pos).sum(dim=1) / self._temp)  # (N,)
        back_sim = torch.exp(torch.matmul(z, z_back.T) / self._temp)  # (N,M)
        back_sim = torch.sum(back_sim, dim=1) + pos_sim  # (N,)
        closs = -torch.log(pos_sim / back_sim)

        return closs.mean()


class TCRContrastiveLoss(Module):
    """
    Contrastive loss designed to work on a mixture of epitope-matched and
    unlabelled background TCRs.
    """

    def __init__(self, temp: float = 0.05) -> None:
        super().__init__()
        self._temp = temp

    def forward(
        self, bg: Tensor, bg_prime: Tensor, ep: Tensor, ep_prime: Tensor
    ) -> Tensor:
        """
        Assumes that all embeddings are all already l2-normalised.
        """
        # N: background batch size, M: labelled batch size
        N = bg.size(0)
        M = ep.size(0)

        # First calculate autocontrastive loss over background
        bg_sim = torch.exp(torch.matmul(bg, bg_prime.T) / self._temp)  # (N,N)
        bg_pos_sim = torch.diag(bg_sim)  # (N,)
        bg_back_sim = torch.sum(bg_sim, dim=1)  # (N,)
        bg_closs = torch.sum(-torch.log(bg_pos_sim / bg_back_sim))  # (1,)

        # then calculate contrastive loss over labelled + background
        ep_pos_sim = torch.exp(torch.sum(ep * ep_prime, dim=1) / self._temp)  # (M,)
        ep_back_sim = (
            torch.sum(torch.exp(torch.matmul(ep, bg_prime.T) / self._temp), dim=1)
            + ep_pos_sim
        )  # (M,)
        ep_closs = torch.sum(-torch.log(ep_pos_sim / ep_back_sim))  # (1,)

        return (bg_closs + ep_closs) / (N + M)
