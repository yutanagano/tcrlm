import pytest
from src.modules import CDR3BERT_c
from src.modules import cdr3bert
import torch


@pytest.fixture
def cdr3bert_c():
    model = CDR3BERT_c(
        num_encoder_layers=6,
        d_model=64,
        nhead=8,
        dim_feedforward=256
    )

    return model


class TestAAEmbedding_c:
    def test_embedding(self):
        embedder = cdr3bert.AAEmbedding_c(embedding_dim=2)
        
        result = embedder(torch.tensor([[[2,1],[3,1],[4,2],[5,2]]]))

        assert result.size() == (1,4,2)


    def test_padding_index(self):
        embedder = cdr3bert.AAEmbedding_c(embedding_dim=2)
        
        padding = embedder(torch.tensor([[[0,0]]]))
        non_padding = embedder(torch.tensor([[[1,1]]]))

        assert (padding == 0).all()
        assert (non_padding != 0).any()


    @pytest.mark.parametrize(
        'token_index', ([-1, 1], [23, 1], [1, -1], [1, 3])
    )
    def test_error_out_of_bounds_token_index(self, token_index):
        embedder = cdr3bert.AAEmbedding_c(embedding_dim=2)
        
        with pytest.raises(IndexError):
            embedder(torch.tensor([[token_index]]))


class TestCDR3BERT_c:
    def test_init_attributes(self, cdr3bert_c):
        assert cdr3bert_c.embed_layer == 5
        assert cdr3bert_c._num_layers == 6
        assert cdr3bert_c._d_model == 64
        assert cdr3bert_c._nhead == 8
        assert cdr3bert_c._dim_feedforward == 256


    def test_forward(self, cdr3bert_c):
        batch = torch.zeros((3,10,3), dtype=torch.long)
        out, padding_mask = cdr3bert_c(x=batch)

        assert out.size() == (3,10,64)
        assert padding_mask.size() == (3,10)
        assert (padding_mask == 1).all()


    def test_embed(self, cdr3bert_c):
        batch = torch.tensor(
            [
                [[3,1,1],[4,1,2],[5,1,3],[3,2,1],[4,2,2],[5,2,3]],
                [[6,1,1],[7,1,2],[8,1,3],[6,2,1],[7,2,2],[8,2,3]],
                [[3,1,1],[4,1,2],[5,1,3],[6,2,1],[7,2,2],[8,2,3]]
            ],
            dtype=torch.long
        )
        out = cdr3bert_c.embed(x=batch)

        assert out.size() == (3,64)
        torch.testing.assert_close(out.norm(dim=1), torch.ones(3))


    def test_mlm(self, cdr3bert_c):
        batch = torch.zeros((3,10,3), dtype=torch.long)
        out = cdr3bert_c.mlm(x=batch)

        assert out.size() == (3,10,20)


    def test_name(self, cdr3bert_c):
        assert cdr3bert_c.name == 'CDR3BERT_c_6_64_8_256-embed_5'