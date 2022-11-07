import pytest
import torch
from torch.testing import assert_close
import src.nn.models as models


@pytest.fixture(scope='module')
def cdr3bert():
    bert = models.Cdr3Bert(
        aa_vocab_size=20,
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

class TestMaskedAveragePool:
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
    def test_masked_average_pool(
        self,
        token_embeddings,
        padding_mask,
        expected
    ):
        result = models.masked_average_pool(
            token_embeddings=token_embeddings,
            padding_mask=padding_mask
        )
        assert_close(result, expected)


class TestAaEmbedder:
    def test_embedding(self):
        embedder = models.AaEmbedder(aa_vocab_size=400, embedding_dim=2)
        
        result = embedder(torch.tensor([[2,1,401,0]]))

        assert result.size() == (1,4,2)


    def test_padding_index(self):
        embedder = models.AaEmbedder(aa_vocab_size=8000, embedding_dim=2)
        
        padding = embedder(torch.tensor([[0]]))
        non_padding = embedder(torch.tensor([[1]]))

        assert (padding == 0).all()
        assert (non_padding != 0).any()


    @pytest.mark.parametrize(
        'token_index', (-1, 22)
    )
    def test_error_out_of_bounds_token_index(self, token_index):
        embedder = models.AaEmbedder(aa_vocab_size=20, embedding_dim=2)
        
        with pytest.raises(IndexError):
            embedder(torch.tensor([[token_index]]))


class TestPositionEncoder:
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
    def test_position_encoder(self, embedding_dim, expected):
        encoder = models.PositionEncoder(
            embedding_dim=embedding_dim,
            dropout=0
        )
        x = torch.zeros((1,3,embedding_dim), dtype=torch.float)
        x = encoder(x)

        assert x.size() == (1,3,embedding_dim)
        assert_close(x, expected)


class TestCdr3Bert:
    def test_init_attributes(self, cdr3bert):
        assert cdr3bert.aa_vocab_size == 20
        assert cdr3bert.embedder.aa_vocab_size == 20
        assert cdr3bert.d_model == 6
        assert cdr3bert.nhead == 2
        assert cdr3bert.dim_feedforward == 48


    def test_forward(self, cdr3bert):
        batch = torch.zeros((3,10), dtype=torch.int)
        out, padding_mask = cdr3bert(x=batch)

        assert out.size() == (3,10,6)
        assert padding_mask.size() == (3,10)
        assert (padding_mask == 1).all()


    def test_embed(self, cdr3bert):
        batch = torch.zeros((3,10), dtype=torch.int)
        out = cdr3bert.embed(x=batch)

        assert out.size() == (3,6)


class TestTcrEmbedder:
    def test_init_attributes(self, tcr_embedder, cdr3bert):
        assert tcr_embedder.d_model == 6
        assert tcr_embedder.alpha_bert == cdr3bert
        assert tcr_embedder.beta_bert == cdr3bert


    def test_forward(self, tcr_embedder):
        alpha_batch = torch.zeros((3,10), dtype=torch.int)
        beta_batch = torch.zeros((3,12), dtype=torch.int)
        out = tcr_embedder(
            x_a=alpha_batch,
            x_b=beta_batch
        )

        assert out.size() == (3,12)


class TestCdr3BertPretrainWrapper:
    def test_init_attributes(self, pretrain_wrapper, cdr3bert):
        assert pretrain_wrapper.bert == cdr3bert
        assert pretrain_wrapper.generator.out_features == 20

    
    def test_forward(self, pretrain_wrapper):
        batch = torch.zeros((3,10), dtype=torch.int)
        out = pretrain_wrapper(x=batch)

        assert out.size() == (3,10,20)


class TestCdr3BertFineTuneWrapper:
    def test_init_attributes(self, finetune_wrapper, tcr_embedder):
        assert finetune_wrapper.d_model == 6
        assert finetune_wrapper.embedder == tcr_embedder


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
        finetune_wrapper.train()
        finetune_wrapper.custom_trainmode()

        assert not finetune_wrapper.embedder.training
        assert finetune_wrapper.classifier.training
