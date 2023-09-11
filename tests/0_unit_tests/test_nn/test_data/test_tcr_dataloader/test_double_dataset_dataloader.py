from pandas import DataFrame
import pytest
import torch
from typing import Iterable

from src.nn.data.tcr_dataset import TcrDataset
from src.nn.data.tcr_dataloader import DoubleDatasetDataLoader
from src.nn.data.batch_collator import BatchCollator
from src.nn.data import schema
from src.nn.data.schema.tcr_pmhc_pair import TcrPmhcPair


BATCH_SIZE_1 = 2
BATCH_SIZE_2 = 3
TCR_FROM_DATASET_1 = schema.make_tcr_from_components("TRAV1-1*01", "CATQYF", "TRBV2*01", "CASQYF")
TCR_FROM_DATASET_2 = schema.make_tcr_from_components("TRAV1-2*01", "CATQYF", "TRBV3-1*01", "CASQYF")


def test_iter(tcr_dataset_1, tcr_dataset_2, no_op_batch_collator):
    dataloader = DoubleDatasetDataLoader(
        dataset_1=tcr_dataset_1, dataset_2=tcr_dataset_2, batch_collator=no_op_batch_collator, device=torch.device("cpu"), batch_size_1=BATCH_SIZE_1, batch_size_2=BATCH_SIZE_2, num_workers_per_dataset=1, distributed=False
    )

    for (batch,) in dataloader:
        assert isinstance(batch, Iterable)
        assert isinstance(batch[0], TcrPmhcPair)
        assert len(batch) == BATCH_SIZE_1 + BATCH_SIZE_2

        for index in range(BATCH_SIZE_1):
            assert batch[index].tcr == TCR_FROM_DATASET_1

        for offset in range(BATCH_SIZE_2):
            assert batch[BATCH_SIZE_1 + offset].tcr == TCR_FROM_DATASET_2


@pytest.fixture
def tcr_dataset_1():
    df = DataFrame(
        {
            "TRAV": ["TRAV1-1*01"] * 4,
            "CDR3A": ["CATQYF"] * 4,
            "TRBV": ["TRBV2*01"] * 4,
            "CDR3B": ["CASQYF"] * 4,
            "Epitope": [None] * 4,
            "MHCA": [None] * 4,
            "MHCB": [None] * 4,
        }
    )
    return TcrDataset(df)


@pytest.fixture
def tcr_dataset_2():
    df = DataFrame(
        {
            "TRAV": ["TRAV1-2*01"] * 9,
            "CDR3A": ["CATQYF"] * 9,
            "TRBV": ["TRBV3-1*01"] * 9,
            "CDR3B": ["CASQYF"] * 9,
            "Epitope": [None] * 9,
            "MHCA": [None] * 9,
            "MHCB": [None] * 9,
        }
    )
    return TcrDataset(df)


@pytest.fixture
def no_op_batch_collator():
    return NoOpCollator(tokeniser=None)


class NoOpCollator(BatchCollator):
    def collate_fn(self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]) -> Iterable[TcrPmhcPair]:
        return (tcr_pmhc_pairs,)