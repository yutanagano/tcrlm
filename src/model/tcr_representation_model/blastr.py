import json
from pathlib import Path
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from pandas import DataFrame

from src.nn.bert import Bert
from src.nn.data.tcr_dataset import TcrDataset
from src.nn.data.tcr_dataloader import SingleDatasetDataLoader
from src.model.tcr_representation_model import TcrRepresentationModel
from src.nn.data.tokeniser.tokeniser import Tokeniser
from src.config_reader import ConfigReader
from src.nn.data.batch_collator import DefaultBatchCollator

from typing import Union


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

    def calc_cdist_matrix(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame
    ) -> ndarray:
        anchor_tcr_representations = self._calc_torch_representations(anchor_tcrs)
        comparison_tcr_representations = self._calc_torch_representations(comparison_tcrs)
        return torch.cdist(anchor_tcr_representations, comparison_tcr_representations, p=2).cpu().numpy()

    def calc_pdist_vector(self, tcrs: DataFrame) -> ndarray:
        tcr_representations = self._calc_torch_representations(tcrs)
        return torch.pdist(tcr_representations, p=2).cpu().numpy()

    def calc_vector_representations(self, tcrs: DataFrame) -> ndarray:
        return self._calc_torch_representations(tcrs).cpu().numpy()

    @torch.no_grad()
    def _calc_torch_representations(self, tcrs: DataFrame) -> Tensor:
        dataloader = self._make_dataloader_for(tcrs)
        batched_representations = [
            self._bert.get_vector_representations_of(batch) for (batch,) in dataloader
        ]
        return torch.concat(batched_representations)

    def _make_dataloader_for(self, tcrs: DataFrame) -> SingleDatasetDataLoader:
        tcrs = tcrs.copy()

        for col in ("Epitope", "MHCA", "MHCB"):
            if col not in tcrs:
                tcrs[col] = None
        
        dataset = TcrDataset(tcrs)
        batch_collator = DefaultBatchCollator(self._tokeniser)
        return SingleDatasetDataLoader(
            dataset,
            batch_collator=batch_collator,
            device=self._device,
            batch_size=512,
            shuffle=False,
        )


def load_blastr_save(path: Path, device: Union[torch.device, str, int, None] = None) -> Blastr:
    with open(path / "config.json", "r") as f:
        config = json.load(f)
    config_reader = ConfigReader(config)

    state_dict = torch.load(path / "state_dict.pt")

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    name = config_reader.get_model_name()
    tokeniser = config_reader.get_tokeniser()
    bert = config_reader.get_bert_on_device(device)
    bert.load_state_dict(state_dict)

    return Blastr(name=name, tokeniser=tokeniser, bert=bert, device=device)
