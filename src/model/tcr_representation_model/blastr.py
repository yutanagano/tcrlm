import json
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
from src.data.batch_collator import DefaultBatchCollator


class Blastr(TcrRepresentationModel):
    name: str = None
    distance_bins = np.linspace(0, 2, num=21)

    def __init__(
        self, name: str, tokeniser: Tokeniser, bert: Bert, device: torch.device
    ) -> None:
        self.name = name
        self._tokeniser = tokeniser
        self._bert = bert.eval()
        self._device = device

    def calc_vector_representations(self, tcrs: DataFrame) -> ndarray:
        tcr_dataloader = self._make_dataloader_for(tcrs)
        bert_representations = self._get_bert_representations_of_tcrs_in(tcr_dataloader)
        bert_representations_as_ndarray = bert_representations.numpy()

        return bert_representations_as_ndarray

    def _make_dataloader_for(self, tcrs: DataFrame) -> TcrDataLoader:
        dataset = TcrDataset(tcrs)
        batch_collator = DefaultBatchCollator(self._tokeniser)
        return TcrDataLoader(dataset, batch_collator=batch_collator, batch_size=512, shuffle=False)

    @torch.no_grad()
    def _get_bert_representations_of_tcrs_in(self, dataloader: TcrDataLoader) -> Tensor:
        batched_representations = [
            self._bert.get_vector_representations_of(batch.to(self._device))
            for batch in dataloader
        ]
        return torch.concat(batched_representations).cpu()

    def calc_cdist_matrix_from_representations(
        self,
        anchor_tcr_representations: ndarray,
        comparison_tcr_representations: ndarray,
    ) -> ndarray:
        return distance.cdist(
            anchor_tcr_representations,
            comparison_tcr_representations,
            metric="euclidean",
        )

    def calc_pdist_vector_from_representations(
        self, tcr_representations: ndarray
    ) -> ndarray:
        return distance.pdist(tcr_representations, metric="euclidean")


def load_blastr_save(path: Path) -> Blastr:
    with open(path / "config.json", "r") as f:
        config = json.load(f)
    config_reader = ConfigReader(config)

    state_dict = torch.load(path / "state_dict.pt")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    name = config_reader.get_model_name()
    tokeniser = config_reader.get_tokeniser()
    bert = config_reader.get_bert_on_device(device)
    bert.load_state_dict(state_dict)

    return Blastr(name=name, tokeniser=tokeniser, bert=bert, device=device)
