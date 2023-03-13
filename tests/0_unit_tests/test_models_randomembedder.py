from src.models import RandomEmbedder


class TestRandomEmbedder:
    def test_name(self):
        embedder = RandomEmbedder(name_idx=420)

        assert embedder.name == 'Random Embedder 420'


    def test_embed(self, abcdr3t_dataloader):
        embedder = RandomEmbedder(dim=5)

        batch = next(iter(abcdr3t_dataloader))

        assert embedder.embed(batch).size() == (3,10)