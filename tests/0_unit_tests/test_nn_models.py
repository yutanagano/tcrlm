import pytest
import torch
import source.nn.models as models


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
    result = models.create_padding_mask(x)
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
    result = models.masked_average_pool(
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
    embedder = models.AaEmbedder(embedding_dim=embedding_dim)
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
    encoder = models.PositionEncoder(
        embedding_dim=embedding_dim,
        dropout=0
    )
    x = torch.zeros((1,3,embedding_dim), dtype=torch.float)
    x = encoder(x)

    assert x.size() == (1,3,embedding_dim)
    assert torch.equal(x, expected)


@pytest.fixture(scope='module')
def cdr3bert():
    bert = models.Cdr3Bert(
        num_encoder_layers=2,
        d_model=6,
        nhead=2,
        dim_feedforward=48
    )
    return bert


@pytest.fixture(scope='module')
def tcr_embedder(cdr3bert):
    embedder = models.TcrEmbedder(
        alpha_bert=cdr3bert,
        beta_bert=cdr3bert
    )
    return embedder


@pytest.fixture(scope='module')
def pretrain_wrapper(cdr3bert):
    wrapper = models.Cdr3BertPretrainWrapper(
        bert=cdr3bert
    )
    return wrapper


@pytest.fixture(scope='module')
def finetune_wrapper(tcr_embedder):
    wrapper = models.Cdr3BertFineTuneWrapper(
        tcr_embedder=tcr_embedder
    )
    wrapper.eval()
    return wrapper


class TestCdr3Bert:
    def test_get_d_model(self, cdr3bert):
        assert cdr3bert.d_model == 6
    

    def test_set_d_model(self, cdr3bert):
        with pytest.raises(AttributeError):
            cdr3bert.d_model = 10


    def test_get_nhead(self, cdr3bert):
        assert cdr3bert.nhead == 2
    

    def test_set_nhead(self, cdr3bert):
        with pytest.raises(AttributeError):
            cdr3bert.nhead = 10


    def test_get_dim_feedforward(self, cdr3bert):
        assert cdr3bert.dim_feedforward == 48
    

    def test_set_dim_feedforward(self, cdr3bert):
        with pytest.raises(AttributeError):
            cdr3bert.dim_feedforward = 10


    def test_forward(self, cdr3bert):
        batch = torch.zeros((3,10), dtype=torch.int)
        out, padding_mask = cdr3bert(x=batch)

        assert out.size() == (3,10,6)
        assert padding_mask.size() == (3,10)


    def test_embed(self, cdr3bert):
        batch = torch.zeros((3,10), dtype=torch.int)
        out = cdr3bert.embed(x=batch)

        assert out.size() == (3,6)


class TestTcrEmbedder:
    def test_get_d_model(self, tcr_embedder):
        assert tcr_embedder.d_model == 6


    def test_set_d_model(self, tcr_embedder):
        with pytest.raises(AttributeError):
            tcr_embedder.d_model = 10
    

    def test_get_alpha_bert(self, tcr_embedder, cdr3bert):
        assert tcr_embedder.alpha_bert == cdr3bert


    def test_set_alpha_bert(self, tcr_embedder):
        with pytest.raises(AttributeError):
            tcr_embedder.alpha_bert = 1
    

    def test_get_beta_bert(self, tcr_embedder, cdr3bert):
        assert tcr_embedder.beta_bert == cdr3bert
    

    def test_set_beta_bert(self, tcr_embedder):
        with pytest.raises(AttributeError):
            tcr_embedder.beta_bert = 1


    def test_forward(self, tcr_embedder):
        alpha_batch = torch.zeros((3,10), dtype=torch.int)
        beta_batch = torch.zeros((3,10), dtype=torch.int)
        out = tcr_embedder(
            x_a=alpha_batch,
            x_b=beta_batch
        )

        assert out.size() == (3,12)


class TestCdr3BertPretrainWrapper:
    def test_get_bert(self, pretrain_wrapper, cdr3bert):
        assert pretrain_wrapper.bert == cdr3bert


    def test_set_bert(self, pretrain_wrapper):
        with pytest.raises(AttributeError):
            pretrain_wrapper.bert = 5

    
    def test_forward(self, pretrain_wrapper):
        batch = torch.zeros((3,10), dtype=torch.int)
        out = pretrain_wrapper(x=batch)

        assert out.size() == (3,10,20)


class TestCdr3BertFineTuneWrapper:
    def test_get_d_model(self, finetune_wrapper):
        assert finetune_wrapper.d_model == 6


    def test_set_d_model(self, finetune_wrapper):
        with pytest.raises(AttributeError):
            finetune_wrapper.d_model = 10


    def test_get_embedder(self, finetune_wrapper, tcr_embedder):
        assert finetune_wrapper.embedder == tcr_embedder


    def test_set_embedder(self, finetune_wrapper):
        with pytest.raises(AttributeError):
            finetune_wrapper.embedder = 1


    def test_forward(self, finetune_wrapper):
        a = torch.zeros((3,10), dtype=torch.int)
        b = torch.zeros((3,15), dtype=torch.int)
        c = torch.zeros((3,12), dtype=torch.int)
        d = torch.zeros((3,12), dtype=torch.int)
        out = finetune_wrapper(
            x_1a=a, x_1b=b,
            x_2a=c, x_2b=d
        )

        assert out.size() == (3,2)
        
    
    def test_custom_trainmode(self, finetune_wrapper):
        finetune_wrapper.custom_trainmode()
        assert not finetune_wrapper._embedder.training
        assert finetune_wrapper.classifier.training
        finetune_wrapper.eval()
