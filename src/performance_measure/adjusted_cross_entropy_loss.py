from torch import FloatTensor, LongTensor
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from libtcrlm.tokeniser.token_indices import DefaultTokenIndex


NUM_INDICES_RESERVED_FOR_DEFAULT_TOKENS = len(DefaultTokenIndex)


class AdjustedCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, label_smoothing: float = 0.1) -> None:
        super().__init__(
            label_smoothing=label_smoothing,
            ignore_index=-NUM_INDICES_RESERVED_FOR_DEFAULT_TOKENS,
        )

    def forward(self, input: FloatTensor, target: LongTensor) -> FloatTensor:
        return F.cross_entropy(
            input=input,
            target=target - NUM_INDICES_RESERVED_FOR_DEFAULT_TOKENS,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
