import pytest
import torch
from source.cdr3bert import masked_average_pool, \
    Cdr3Bert, Cdr3BertPretrainWrapper


@pytest.fixture(scope='module')
def instantiate_bert():
    bert = Cdr3Bert(num_encoder_layers=2,
                    d_model=6,
                    nhead=2,
                    dim_feedforward=48)
    yield bert


# Positive tests
@pytest.mark.parametrize(
    ('batch_list','mask_list','expected_list'),
    (
        (
            [
                [[5,0,2],[4,6,2],[7,3,5]],
                [[3,4,1],[9,7,2],[7,8,6]]
            ],
            [
                [1,1,0],[1,0,0]
            ],
            [
                [4.5,3,2],[3,4,1]
            ]
        ),
        (
            [
                [[3,7,1],[5,6,1],[7,2,1]],
                [[3,4,1],[9,8,2],[0,8,6]]
            ],
            [
                [1,1,1],[1,1,0]
            ],
            [
                [5,5,1],[6,6,1.5]
            ]
        )
    )
)
def test_masked_average_pool(batch_list,mask_list,expected_list):
    batch = torch.tensor(batch_list, dtype=torch.int)
    mask = torch.tensor(mask_list, dtype=torch.int)
    expected = torch.tensor(expected_list, dtype=torch.float32)

    result = masked_average_pool(batch, mask)

    assert(torch.equal(result, expected))


def test_forward(instantiate_bert):
    bert = instantiate_bert
    batch = torch.zeros((3,10), dtype=torch.int)
    out, padding_mask = bert(x=batch)

    assert(out.size() == (3,10,6))
    assert(padding_mask.size() == (3,10))


def test_fill_in(instantiate_bert):
    bert = instantiate_bert
    batch = torch.zeros((3,10), dtype=torch.int)
    out = bert.fill_in(x=batch)

    assert(out.size() == (3,10,20))


def test_embed(instantiate_bert):
    bert = instantiate_bert
    batch = torch.zeros((3,10), dtype=torch.int)
    out = bert.embed(x=batch)

    assert(out.size() == (3,6))


def test_pretrain_wrapper(instantiate_bert):
    bert = instantiate_bert
    pretrain_bert = Cdr3BertPretrainWrapper(bert)
    batch = torch.zeros((3,10), dtype=torch.int)
    out = pretrain_bert(x=batch)

    assert(out.size() == (3,10,20))