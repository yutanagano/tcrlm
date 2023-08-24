from pathlib import Path
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from scipy.spatial import distance

from src.model.bert import Bert
from src.data.tcr_dataset import TcrDataset
from src.data.tcr_dataloader import TcrDataLoader
from src.model.tcr_representation_model import TcrRepresentationModel
from src.data.tokeniser.tokeniser import Tokeniser
from src.model_trainer.config_reader import ConfigReader


class Blastr(TcrRepresentationModel):
    name: str = None
    distance_bins = np.linspace(0, 2, num=21)

    def __init__(self, name: str, tokeniser: Tokeniser, bert: Bert) -> None:
        self.name = name
        self.tokeniser = tokeniser
        self.bert = bert

    def calc_cdist_matrix(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame
    ) -> ndarray:
        anchor_representations = self.calc_vector_representations(anchor_tcrs)
        comparison_representations = self.calc_vector_representations(comparison_tcrs)

        return distance.cdist(
            anchor_representations, comparison_representations, metric="euclidean"
        )

    def calc_pdist_vector(self, tcrs: DataFrame) -> ndarray:
        tcr_representations = self.calc_vector_representations(tcrs)
        return distance.pdist(tcr_representations, metric="euclidean")

    def calc_vector_representations(self, tcrs: DataFrame) -> ndarray:
        tcr_dataloader = self._make_dataloader_for(tcrs)
        bert_representations = self._get_bert_representations_of_tcrs_in(tcr_dataloader)
        bert_representations_as_ndarray = bert_representations.numpy()

        return bert_representations_as_ndarray

    def _make_dataloader_for(self, tcrs: DataFrame) -> TcrDataLoader:
        dataset = TcrDataset(tcrs, tokeniser=self.tokeniser)
        return TcrDataLoader(dataset, batch_size=512, shuffle=False)

    @torch.no_grad()
    def _get_bert_representations_of_tcrs_in(self, dataloader: TcrDataLoader) -> Tensor:
        batched_representations = [
            self.bert.get_vector_representations_of(batch) for batch in dataloader
        ]
        return torch.concat(batched_representations)


def load_blastr_save(path: Path) -> Blastr:
    config_reader = ConfigReader(path/"config.json")
    state_dict = torch.load(path/"state_dict.pt")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    name = config_reader.get_model_name()
    tokeniser = config_reader.get_tokeniser()
    bert = config_reader.get_bert_on_device(device)
    bert.load_state_dict(state_dict)

    return Blastr(name=name, tokeniser=tokeniser, bert=bert)