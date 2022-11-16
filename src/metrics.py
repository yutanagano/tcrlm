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


class AdjustedCELoss(CrossEntropyLoss):
    '''
    Custom cross entropy loss class which subtracts 2 from the input labels
    before running the cross entropy funciton. This is because our models'
    vocabulary space is indexed from 0...X whereas the target data's vocabulary
    is indexed from 2...X+2 (index 0 and 1 are reserved for the padding and
    masked tokens respectively).
    '''

    def __init__( self, label_smoothing: float = 0 ) -> None:
        super().__init__(label_smoothing=label_smoothing, ignore_index=-2)
        # ignore_index is set to -2 here because the padding token, indexed at
        # 0 in the target vocabulary space, will be mapped to -2 when
        # moving all indices two places down.

    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            input,
            target-2,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )