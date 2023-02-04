from src.modules import RandomEmbedder


class TestRandomEmbedder:
    def test_embed(self, abcdr3t_dataloader):
        embedder = RandomEmbedder(dim=5)

        batch = next(iter(abcdr3t_dataloader))

        assert embedder.embed(batch).size() == (3,10)