import pandas as pd
import pytest
from src.datahandling import dataloaders
import torch
from torch.utils.data import BatchSampler, RandomSampler


class TestTCRDataLoader:
    def test_init(self, abcdr3t_dataset):
        dataloader = dataloaders.TCRDataLoader(
            dataset=abcdr3t_dataset, batch_size=3, num_workers=3
        )

        assert dataloader.dataset == abcdr3t_dataset
        assert dataloader.batch_size == 3
        assert type(dataloader.sampler) == RandomSampler
        assert dataloader.sampler.data_source == abcdr3t_dataset
        assert type(dataloader.batch_sampler) == BatchSampler
        assert dataloader.batch_sampler.sampler == dataloader.sampler
        assert dataloader.batch_sampler.batch_size == 3
        assert dataloader.num_workers == 3

    def test_padding_collation(self, abcdr3t_dataset):
        dataloader = dataloaders.TCRDataLoader(
            dataset=abcdr3t_dataset, batch_size=3, shuffle=False
        )

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
        dataloader = dataloaders.MLMDataLoader(dataset=abcdr3t_dataset, batch_size=3)

        masked, target = next(iter(dataloader))

        assert type(masked) == type(target) == torch.Tensor
        assert masked.dim() == 3
        assert target.dim() == 2
        assert masked.size(0) == target.size(0) == 3
        assert masked.size(1) == target.size(1) == 12
        assert masked.size(2) == 4


class TestAutoContrastiveDataLoader:
    def test_shapes(self, abcdr3t_auto_contrastive_dataset):
        dataloader = dataloaders.AutoContrastiveDataLoader(
            dataset=abcdr3t_auto_contrastive_dataset, batch_size=3
        )

        x, x_prime, masked, target = next(iter(dataloader))

        assert type(x) == type(x_prime) == type(masked) == type(target) == torch.Tensor
        assert x.dim() == x_prime.dim() == masked.dim() == 3
        assert target.dim() == 2
        assert x.size(0) == x_prime.size(0) == masked.size(0) == target.size(0) == 3
        assert x.size(1) == masked.size(1) == target.size(1) == 12
        assert x_prime.size(1) in (6, 7, 12)
        assert x.size(2) == x_prime.size(2) == masked.size(2) == 4


class TestEpitopeAutoContrastiveSuperDataLoader:
    def test_shapes(
        self, abcdr3t_auto_contrastive_dataset, abcdr3t_epitope_contrastive_dataset
    ):
        dataloader = dataloaders.EpitopeAutoContrastiveSuperDataLoader(
            dataset_ac=abcdr3t_auto_contrastive_dataset,
            dataset_ec=abcdr3t_epitope_contrastive_dataset,
            batch_size=3,
            num_workers_ac=3,
            num_workers_ec=3,
        )

        ac, ac_prime, ac_masked, ac_target, ec, ec_prime = next(iter(dataloader))

        assert (
            type(ac)
            == type(ac_prime)
            == type(ac_masked)
            == type(ac_target)
            == type(ec)
            == type(ec_prime)
            == torch.Tensor
        )

        assert (
            ac.dim()
            == ac_prime.dim()
            == ac_masked.dim()
            == ec.dim()
            == ec_prime.dim()
            == 3
        )
        assert ac_target.dim() == 2

        assert (
            ac.size(0)
            == ac_prime.size(0)
            == ac_masked.size(0)
            == ac_target.size(0)
            == ec.size(0)
            == ec_prime.size(0)
            == 3
        )
        assert ac.size(1) == ac_masked.size(1) == ac_target.size(1) == ec.size(1) == 12
        assert ac_prime.size(1) in (6, 7, 12)
        assert ec_prime.size(1) in (7, 12)
        assert (
            ac.size(2)
            == ac_prime.size(2)
            == ac_masked.size(2)
            == ec.size(2)
            == ec_prime.size(2)
            == 4
        )

    def test_len(
        self, abcdr3t_auto_contrastive_dataset, abcdr3t_epitope_contrastive_dataset
    ):
        orig_data = abcdr3t_auto_contrastive_dataset._data
        abcdr3t_auto_contrastive_dataset._data = pd.concat((orig_data, orig_data))
        dataloader = dataloaders.EpitopeAutoContrastiveSuperDataLoader(
            dataset_ac=abcdr3t_auto_contrastive_dataset,
            dataset_ec=abcdr3t_epitope_contrastive_dataset,
            batch_size=3,
            num_workers_ac=3,
            num_workers_ec=3,
        )

        assert len(dataloader) == 2

    def test_iter(
        self, abcdr3t_auto_contrastive_dataset, abcdr3t_epitope_contrastive_dataset
    ):
        dataloader = dataloaders.EpitopeAutoContrastiveSuperDataLoader(
            dataset_ac=abcdr3t_auto_contrastive_dataset,
            dataset_ec=abcdr3t_epitope_contrastive_dataset,
            batch_size=1,
            num_workers_ac=3,
            num_workers_ec=3,
        )

        iterations = 0

        for _ in dataloader:
            iterations += 1

        assert iterations == 3

    @pytest.mark.parametrize("to_extend", ("a", "e"))
    def test_wrap_around(
        self,
        abcdr3t_auto_contrastive_dataset,
        abcdr3t_epitope_contrastive_dataset,
        to_extend,
    ):
        if to_extend == "a":
            ds_to_extend = abcdr3t_auto_contrastive_dataset
        elif to_extend == "e":
            ds_to_extend = abcdr3t_epitope_contrastive_dataset

        orig_data = ds_to_extend._data
        ds_to_extend._data = pd.concat((orig_data, orig_data))

        dataloader = dataloaders.EpitopeAutoContrastiveSuperDataLoader(
            dataset_ac=abcdr3t_auto_contrastive_dataset,
            dataset_ec=abcdr3t_epitope_contrastive_dataset,
            batch_size=1,
            num_workers_ac=3,
            num_workers_ec=3,
        )

        assert len(dataloader) == 6

        iterations = 0
        for _ in dataloader:
            iterations += 1
