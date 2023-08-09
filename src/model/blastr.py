import torch
from torch import Tensor
from typing import Iterable
from numpy import ndarray

from src.tcr import Tcr
from src.model.bert import Bert
from src.data.tcr_dataset import TcrDataset
from src.data.tcr_dataloader import TcrDataLoader
from src.model.tcr_representation_model import TcrRepresentationModel
from src.data.tokeniser.tokeniser import Tokeniser


class Blastr(TcrRepresentationModel):
    name: str = None

    def __init__(self, name: str, tokeniser: Tokeniser, bert: Bert) -> None:
        self.name = name
        self.tokeniser = tokeniser
        self.bert = bert
        
    def calc_representations_of(self, tcrs: Iterable[Tcr]) -> ndarray:
        tcr_dataloader = self._make_dataloader_for(tcrs)
        bert_representations = self._get_bert_representations_of_tcrs_in(tcr_dataloader)
        bert_representations_as_ndarray = bert_representations.numpy()

        return bert_representations_as_ndarray
    
    def _make_dataloader_for(self, tcrs: Iterable[Tcr]) -> TcrDataLoader:
        dataset = TcrDataset(tcrs, tokeniser=self.tokeniser)
        return TcrDataLoader(dataset, batch_size=512, shuffle=False)

    @torch.no_grad
    def _get_bert_representations_of_tcrs_in(self, dataloader: TcrDataLoader) -> Tensor:
        batched_representations = [self.bert.get_vector_representations_of(batch) for batch in dataloader]
        return torch.concat(batched_representations)