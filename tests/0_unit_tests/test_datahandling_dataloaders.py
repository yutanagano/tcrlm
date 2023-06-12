from src.datahandling.dataloaders import *
import torch
from torch.utils.data import BatchSampler, SequentialSampler


class TestTCRDataLoader:
    def test_init(self, abcdr3t_dataset):
        dataloader = TCRDataLoader(dataset=abcdr3t_dataset, batch_size=3, num_workers=3)

        assert dataloader.dataset == abcdr3t_dataset
        assert dataloader.batch_size == 3
        assert type(dataloader.sampler) == SequentialSampler
        assert dataloader.sampler.data_source == abcdr3t_dataset
        assert type(dataloader.batch_sampler) == BatchSampler
        assert dataloader.batch_sampler.sampler == dataloader.sampler
        assert dataloader.batch_sampler.batch_size == 3
        assert dataloader.num_workers == 3

    def test_padding_collation(self, abcdr3t_dataset):
        dataloader = TCRDataLoader(dataset=abcdr3t_dataset, batch_size=3, shuffle=False)

        expected = torch.tensor(
            [
                [
                    [2, 0, 0, 0],
                    [4, 1, 6, 1],
                    [3, 2, 6, 1],
                    [18, 3, 6, 1],
                    [16, 4, 6, 1],
                    [22, 5, 6, 1],
                    [7, 6, 6, 1],
                    [4, 1, 5, 2],
                    [3, 2, 5, 2],
                    [19, 3, 5, 2],
                    [22, 4, 5, 2],
                    [21, 5, 5, 2],
                ],
                [
                    [2, 0, 0, 0],
                    [4, 1, 6, 1],
                    [3, 2, 6, 1],
                    [18, 3, 6, 1],
                    [16, 4, 6, 1],
                    [22, 5, 6, 1],
                    [7, 6, 6, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [2, 0, 0, 0],
                    [4, 1, 5, 2],
                    [3, 2, 5, 2],
                    [19, 3, 5, 2],
                    [22, 4, 5, 2],
                    [21, 5, 5, 2],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            ]
        )

        first_batch = next(iter(dataloader))

        assert first_batch.equal(expected)


class TestMLMDataLoader:
    def test_shapes(self, abcdr3t_dataset):
        dataloader = MLMDataLoader(dataset=abcdr3t_dataset, batch_size=3)

        masked, target = next(iter(dataloader))

        assert type(masked) == type(target) == torch.Tensor
        assert masked.dim() == 3
        assert target.dim() == 2
        assert masked.size(0) == target.size(0) == 3
        assert masked.size(1) == target.size(1) == 12
        assert masked.size(2) == 4


class TestAutoContrastiveDataLoader:
    def test_shapes(self, abcdr3t_contrastive_dataset):
        dataloader = ContrastiveDataLoader(
            dataset=abcdr3t_contrastive_dataset, batch_size=3
        )

        x, x_prime, masked, target = next(iter(dataloader))

        assert type(x) == type(x_prime) == type(masked) == type(target) == torch.Tensor
        assert x.dim() == x_prime.dim() == masked.dim() == 3
        assert target.dim() == 2
        assert x.size(0) == x_prime.size(0) == masked.size(0) == target.size(0) == 3
        assert x.size(1) == masked.size(1) == target.size(1) == 12
        assert x_prime.size(1) in (6, 7, 12)
        assert x.size(2) == x_prime.size(2) == masked.size(2) == 4


class TestEpitopeContrastiveDataLoader_dep:
    def test_shapes(self, abcdr3t_epitope_contrastive_dataset):
        dataloader = EpitopeContrastiveDataLoader_dep(
            dataset=abcdr3t_epitope_contrastive_dataset, batch_size=2
        )

        assert len(dataloader) == 1

        x, x_prime, masked, target = next(iter(dataloader))

        assert type(x) == type(x_prime) == type(masked) == type(target) == torch.Tensor
        assert x.dim() == x_prime.dim() == masked.dim() == 3
        assert target.dim() == 2
        assert x.size(0) == x_prime.size(0) == masked.size(0) == target.size(0) == 4
        assert x.size(1) == masked.size(1) == target.size(1) == 12
        assert x_prime.size(1) in (6, 7, 12)
        assert x.size(2) == x_prime.size(2) == masked.size(2) == 4


class TestCombinedContrastiveIterator:
    def test_batching(
        self, abcdr3t_contrastive_dataset
    ):
        dataloader_ac = ContrastiveDataLoader(
            dataset=abcdr3t_contrastive_dataset, batch_size=1
        )
        dataloader_ec = ContrastiveDataLoader(
            dataset=abcdr3t_contrastive_dataset, batch_size=1
        )

        combined_iterator = CombinedContrastiveIterator(
            dataloader_ac=dataloader_ac, dataloader_ec=dataloader_ec
        )

        bg, bg_prime, masked, target, ep, ep_prime, _, _ = next(iter(combined_iterator))

        assert (
            type(bg)
            == type(bg_prime)
            == type(masked)
            == type(target)
            == type(ep)
            == type(ep_prime)
            == torch.Tensor
        )

    def test_iteration(
        self, abcdr3t_contrastive_dataset
    ):
        dataloader_ac = ContrastiveDataLoader(
            dataset=abcdr3t_contrastive_dataset, batch_size=1
        )
        dataloader_ec = ContrastiveDataLoader(
            dataset=abcdr3t_contrastive_dataset, batch_size=1
        )

        combined_iterator = CombinedContrastiveIterator(
            dataloader_ac=dataloader_ac, dataloader_ec=dataloader_ec
        )

        assert len(combined_iterator) == 3
        batches = 0

        for _ in combined_iterator:
            batches += 1

        assert batches == 3
