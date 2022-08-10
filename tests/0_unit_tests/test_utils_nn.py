import pytest
import source.utils.nn as nnutils
import torch


@pytest.mark.parametrize(
    ('x', 'expected'),
    (
        (
            torch.tensor([[1,2,3,21,21]],dtype=torch.long),
            torch.tensor([[False,False,False,True,True]],dtype=torch.bool)
        ),
        (
            torch.tensor([[1,21,21,21,21]],dtype=torch.long),
            torch.tensor([[False,True,True,True,True]],dtype=torch.bool)
        ),
        (
            torch.tensor([[1,2,3,4,21]],dtype=torch.long),
            torch.tensor([[False,False,False,False,True]],dtype=torch.bool)
        )
    )
)
def test_create_padding_mask(x, expected):
    result = nnutils.create_padding_mask(x)
    assert torch.equal(result, expected)


@pytest.mark.parametrize(
    ('token_embeddings','padding_mask','expected'),
    (
        (
            torch.tensor(
                [
                    [[5,0,2],[4,6,2],[7,3,5]],
                    [[3,4,1],[9,7,2],[7,8,6]]
                ],
                dtype=torch.int
            ),
            torch.tensor(
                [
                    [0,0,1],[0,1,1]
                ],
                dtype=torch.int
            ),
            torch.tensor(
                [
                    [4.5,3,2],[3,4,1]
                ],
                dtype=torch.float32
            )
        ),
        (
            torch.tensor(
                [
                    [[3,7,1],[5,6,1],[7,2,1]],
                    [[3,4,1],[9,8,2],[0,8,6]]
                ],
                dtype=torch.int
            ),
            torch.tensor(
                [
                    [0,0,0],[0,0,1]
                ],
                dtype=torch.int
            ),
            torch.tensor(
                [
                    [5,5,1],[6,6,1.5]
                ],
                dtype=torch.float32
            )
        )
    )
)
def test_masked_average_pool(token_embeddings,padding_mask,expected):
    result = nnutils.masked_average_pool(
        token_embeddings=token_embeddings,
        padding_mask=padding_mask
    )
    assert torch.equal(result, expected)


@pytest.mark.parametrize(
    ('embedding_dim', 'x', 'expected_size'),
    (
        (
            6,
            torch.tensor(
                [
                    [1,2,3,21,21,21],
                    [1,2,21,21,21,21],
                    [1,0,3,0,5,21]
                ],
                dtype=torch.int
            ),
            (3,6,6)
        ),
        (
            8,
            torch.tensor(
                [
                    [1,2,3,4,21],
                    [1,2,21,21,21],
                    [1,0,3,0,5]
                ],
                dtype=torch.int
            ),
            (3,5,8)
        ),
    )
)
def test_embedding(embedding_dim, x, expected_size):
    embedder = nnutils.AaEmbedder(embedding_dim=embedding_dim)
    result = embedder(x)

    assert result.size() == expected_size


@pytest.mark.parametrize(
    ('embedding_dim', 'expected'),
    (
        (
            2,
            torch.stack(
                (
                    torch.sin(torch.tensor([0,1,2])),
                    torch.cos(torch.tensor([0,1,2]))
                ),
                dim=0
            ).t().unsqueeze(0)
        ),
        (
            4,
            torch.stack(
                (
                    torch.sin(torch.tensor([0,1,2])),
                    torch.cos(torch.tensor([0,1,2])),
                    torch.sin(torch.tensor([0,1,2]) / (30 ** (2 / 4))),
                    torch.cos(torch.tensor([0,1,2]) / (30 ** (2 / 4))),
                ),
                dim=0
            ).t().unsqueeze(0)
        )
    )
)
def test_position_encoder(embedding_dim, expected):
    encoder = nnutils.PositionEncoder(
        embedding_dim=embedding_dim,
        dropout=0
    )
    x = torch.zeros((1,3,embedding_dim), dtype=torch.float)
    x = encoder(x)

    assert x.size() == (1,3,embedding_dim)
    assert torch.equal(x, expected)