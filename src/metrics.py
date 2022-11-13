import torch
from torch import Tensor


def alignment(x: Tensor, labels: Tensor, alpha: int = 2):
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
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()