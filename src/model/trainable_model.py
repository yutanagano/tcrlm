from torch import Tensor
from torch.nn import Module
from typing import Tuple

from src.model.bert import Bert


class TrainableModel(Module):
    def __init__(self, bert: Bert) -> None:
        super().__init__()
        self.bert = bert


class MlmTrainableModel(TrainableModel):
    def forward(self, tokenised_and_masked_tcrs: Tensor) -> Tensor:
        return self.bert.get_mlm_token_predictions_for(tokenised_and_masked_tcrs)


class ClTrainableModel(TrainableModel):
    def forward(self, tokenised_anchor_tcrs: Tensor, tokenised_positive_pair_tcrs: Tensor, tokenised_and_masked_tcrs: Tensor) -> Tuple[Tensor]:
        anchor_tcr_representations = self.bert.get_vector_representations_of(tokenised_anchor_tcrs)
        positive_pair_tcr_representations = self.bert.get_vector_representations_of(tokenised_positive_pair_tcrs)
        mlm_logits = self.bert.get_mlm_token_predictions_for(tokenised_and_masked_tcrs)

        return anchor_tcr_representations, positive_pair_tcr_representations, mlm_logits
