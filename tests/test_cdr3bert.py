import pytest
import torch
import source.nn.models as models
from source.utils.nn import masked_average_pool


@pytest.fixture(scope='module')
def instantiate_bert():
    bert = models.Cdr3Bert(num_encoder_layers=2,
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
                [0,0,1],[0,1,1]
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
                [0,0,0],[0,0,1]
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


def test_bert_get_d_model(instantiate_bert):
    bert = instantiate_bert
    assert(bert.d_model == 6)


def test_bert_get_nhead(instantiate_bert):
    bert = instantiate_bert
    assert(bert.nhead == 2)


def test_bert_get_dim_feedforward(instantiate_bert):
    bert = instantiate_bert
    assert(bert.dim_feedforward == 48)


def test_bert_forward(instantiate_bert):
    bert = instantiate_bert
    batch = torch.zeros((3,10), dtype=torch.int)
    out, padding_mask = bert(x=batch)

    assert(out.size() == (3,10,6))
    assert(padding_mask.size() == (3,10))


def test_bert_embed(instantiate_bert):
    bert = instantiate_bert
    batch = torch.zeros((3,10), dtype=torch.int)
    out = bert.embed(x=batch)

    assert(out.size() == (3,6))


def test_tcr_embedder(instantiate_bert):
    bert = instantiate_bert
    embedder = models.TcrEmbedder(bert, bert)

    assert(embedder.d_model == 6)
    assert(embedder.alpha_bert == bert)
    assert(embedder.beta_bert == bert)

    alpha_batch = torch.zeros((3,10), dtype=torch.int)
    beta_batch = torch.zeros((3,10), dtype=torch.int)
    out = embedder(alpha_batch, beta_batch)

    assert(out.size() == (3,12))


def test_pretrain_wrapper(instantiate_bert):
    bert = instantiate_bert
    pretrain_bert = models.Cdr3BertPretrainWrapper(bert)
    batch = torch.zeros((3,10), dtype=torch.int)
    out = pretrain_bert(x=batch)

    assert(out.size() == (3,10,20))
    assert(type(pretrain_bert.bert) == models.Cdr3Bert)

    with pytest.raises(AttributeError):
        pretrain_bert.bert = 5


def test_fine_tune_wrapper(instantiate_bert):
    bert = instantiate_bert
    embedder = models.TcrEmbedder(bert, bert)
    finetune_bert = models.Cdr3BertFineTuneWrapper(embedder)

    a = torch.zeros((3,10), dtype=torch.int)
    b = torch.zeros((3,15), dtype=torch.int)
    c = torch.zeros((3,12), dtype=torch.int)
    d = torch.zeros((3,12), dtype=torch.int)
    out = finetune_bert(
        x_1a=a, x_1b=b,
        x_2a=c, x_2b=d
    )

    assert(out.size() == (3,2))
    assert(finetune_bert.embedder == embedder)

    with pytest.raises(AttributeError):
        finetune_bert.d_model = 5
    
    with pytest.raises(AttributeError):
        finetune_bert.embedder = 5
    
    finetune_bert.custom_trainmode()
    assert(not finetune_bert._embedder.training)
    assert(finetune_bert.classifier.training)


# Negative tests
def test_bert_set_d_model(instantiate_bert):
    bert = instantiate_bert
    with pytest.raises(AttributeError):
        bert.d_model = 10


def test_bert_set_nhead(instantiate_bert):
    bert = instantiate_bert
    with pytest.raises(AttributeError):
        bert.nhead = 10


def test_bert_set_dim_feedforward(instantiate_bert):
    bert = instantiate_bert
    with pytest.raises(AttributeError):
        bert.dim_feedforward = 10