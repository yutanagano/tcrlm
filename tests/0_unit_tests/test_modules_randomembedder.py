import pytest
from src.datahandling.dataloaders import TCRDataLoader
from src.modules import RandomEmbedder


@pytest.fixture
def cdr3t_dataloader(cdr3t_dataset):
    dl = TCRDataLoader(dataset=cdr3t_dataset, batch_size=3)
    return dl


class TestRandomEmbedder:
    def test_embed(self, cdr3t_dataloader):
        embedder = RandomEmbedder(dim=5)

        batch = next(iter(cdr3t_dataloader))

        assert embedder.embed(batch).size() == (3,10)


    def test_name(self):
        embedder = RandomEmbedder(name_idx=7)

        assert embedder.name == 'random_embedder_7'