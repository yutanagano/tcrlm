import pytest
from torch import Tensor
from torch.nn.utils import rnn

from src.data.tokeniser.cdr_tokeniser import CdrTokeniser
from src.data.tcr_dataset import TcrDataset
from src.data.tcr_dataloader import TcrDataLoader
from src.data.tokeniser.token_indices import DefaultTokenIndex

BATCH_SIZE = 3
MAX_TOKENISED_TCR_LENGTH = 48
TOKEN_NUM_DIMS = 4


def test_iter(tcr_dataset):
    dataloader = TcrDataLoader(
        tcr_dataset, batch_size=BATCH_SIZE, collate_fn=demo_collate_fn
    )

    for batch in dataloader:
        assert type(batch) == Tensor
        assert batch.dim() == 3
        assert batch.size(0) == BATCH_SIZE
        assert batch.size(1) == MAX_TOKENISED_TCR_LENGTH
        assert batch.size(2) == TOKEN_NUM_DIMS


def test_set_epoch(tcr_dataset):
    mock_sampler = MockSampler()
    dataloader = TcrDataLoader(tcr_dataset, sampler=mock_sampler)
    EPOCH = 420

    dataloader.set_epoch(EPOCH)

    assert mock_sampler.epoch_set_as == EPOCH


class MockSampler:
    def __init__(self) -> None:
        self.epoch_set_as = None

    def set_epoch(self, epoch: int):
        self.epoch_set_as = epoch


@pytest.fixture
def tcr_dataset(mock_data_df):
    return TcrDataset(mock_data_df, CdrTokeniser())


def demo_collate_fn(batch: Tensor) -> Tensor:
    return rnn.pad_sequence(
        batch, batch_first=True, padding_value=DefaultTokenIndex.NULL
    )
