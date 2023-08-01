import torch
from torch import Tensor
from typing import Iterable
import numpy as np
from numpy import ndarray

from src.tcr import Tcr
from src.model.bert import Bert
from src.model.data.tcr_dataset import TcrDataset
from src.model.data.tcr_dataloader import TcrDataLoader
from src.model.tcr_metric import TcrMetric
from src.model.tcr_representation_model import TcrRepresentationModel
from src.model.data.tokenisers.tokeniser import Tokeniser


class Blastr(TcrMetric, TcrRepresentationModel):
    name: str = None

    def __init__(self, name: str, tokeniser: Tokeniser, bert: Bert) -> None:
        self.name = name
        self.tokeniser = tokeniser
        self.bert = bert
    
    def calc_distance_between(self, anchor: Tcr, comparison: Tcr) -> float:
        anchor_representation = self.calc_representation_of(anchor)
        comparison_representation = self.calc_representation_of(comparison)
        difference = anchor_representation - comparison_representation

        return np.linalg.norm(difference, ord=2)

    def calc_cdist_matrix(self, anchors: Iterable[Tcr], comparisons: Iterable[Tcr]) -> ndarray:
        anchor_representations = self._calc_torch_tensor_representations_of(anchors)
        comparison_representations = self._calc_torch_tensor_representations_of(comparisons)
        cdist_matrix = torch.cdist(anchor_representations, comparison_representations, p=2)
        cdist_matrix_as_ndarray = cdist_matrix.numpy()

        return cdist_matrix_as_ndarray
    
    def calc_pdist_vector(self, tcrs: Iterable[Tcr]) -> ndarray:
        tcr_representations = self._calc_torch_tensor_representations_of(tcrs)
        pdist_vector = torch.pdist(tcr_representations, p=2)
        pdist_vector_as_ndarray = pdist_vector.numpy()

        return pdist_vector_as_ndarray
    
    def calc_representation_of(self, tcr: Tcr) -> ndarray:
        return self.calc_representations_of([tcr])[0]
    
    def calc_representations_of(self, tcrs: Iterable[Tcr]) -> ndarray:
        bert_representations = self._calc_torch_tensor_representations_of(tcrs)
        bert_representations_as_ndarray = bert_representations.numpy()

        return bert_representations_as_ndarray
    
    def _calc_torch_tensor_representations_of(self, tcrs: Iterable[Tcr]) -> Tensor:
        tcr_dataloader = self._make_dataloader_for(tcrs)
        bert_representations = self._get_bert_representations_of_tcrs_in(tcr_dataloader)

        return bert_representations
    
    def _make_dataloader_for(self, tcrs: Iterable[Tcr]):
        dataset = TcrDataset(tcrs, tokeniser=self.tokeniser)
        return TcrDataLoader(dataset, batch_size=512, shuffle=False)

    @torch.no_grad
    def _get_bert_representations_of_tcrs_in(self, dataloader):
        batched_representations = [self.bert.get_vector_representations_of(batch) for batch in dataloader]
        return torch.concat(batched_representations)