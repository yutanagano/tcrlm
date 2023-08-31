import pytest
import torch
from torch import Tensor

from src.nn.data.tokeniser import BetaCdr3Tokeniser
from src.nn.token_embedder import BetaCdr3SimpleEmbedder


def test_forward(mock_batch_of_tokenised_tcrs):
    BATCH_DIMENSIONALITY = 3
    BATCH_SIZE = 2

    embedder = BetaCdr3SimpleEmbedder()
    batch_of_embedded_mock_tcrs = embedder.forward(mock_batch_of_tokenised_tcrs)
    embedded_mock_tcr = batch_of_embedded_mock_tcrs[0]
    expected_embedded_mock_tcr = torch.tensor(
        [
            [
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.2,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0.4,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0.6,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0.8,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
        ]
    )

    assert type(batch_of_embedded_mock_tcrs) == Tensor

    assert batch_of_embedded_mock_tcrs.dim() == BATCH_DIMENSIONALITY
    assert batch_of_embedded_mock_tcrs.size(0) == BATCH_SIZE

    assert torch.equal(embedded_mock_tcr, expected_embedded_mock_tcr)


@pytest.fixture
def mock_batch_of_tokenised_tcrs(mock_tcr):
    tokeniser = BetaCdr3Tokeniser()
    tokenised_mock_tcr = tokeniser.tokenise(mock_tcr)
    batch_of_tokenised_mock_tcrs = torch.stack([tokenised_mock_tcr, tokenised_mock_tcr])

    return batch_of_tokenised_mock_tcrs
