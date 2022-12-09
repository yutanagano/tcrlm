from src.datahandling import dataloaders
import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler


class TestTCRDataLoader:
    def test_init(self, cdr3t_dataset):
        dataloader = dataloaders.TCRDataLoader(
            dataset=cdr3t_dataset,
            batch_size=3,
            num_workers=3
        )

        assert dataloader.dataset == cdr3t_dataset
        assert dataloader.batch_size == 3
        assert type(dataloader.sampler) == RandomSampler
        assert dataloader.sampler.data_source == cdr3t_dataset
        assert type(dataloader.batch_sampler) == BatchSampler
        assert dataloader.batch_sampler.sampler == dataloader.sampler
        assert dataloader.batch_sampler.batch_size == 3
        assert dataloader.num_workers == 3


    def test_padding_collation(self, cdr3t_dataset):
        dataloader = dataloaders.TCRDataLoader(
            dataset=cdr3t_dataset,
            batch_size=3,
            shuffle=False
        )

        expected = torch.tensor(
            [
                [
                    [2,0,0],
                    [4,1,1],[3,1,2],[18,1,3],[16,1,4],[22,1,5],[7,1,6],
                    [4,2,1],[3,2,2],[19,2,3],[22,2,4],[21,2,5]
                ],
                [
                    [2,0,0],
                    [4,1,1],[3,1,2],[18,1,3],[16,1,4],[22,1,5],[7,1,6],
                    [0,0,0],[0,0,0],[ 0,0,0],[ 0,0,0],[ 0,0,0]
                ],
                [
                    [2,0,0],
                    [4,2,1],[3,2,2],[19,2,3],[22,2,4],[21,2,5],
                    [0,0,0],[0,0,0],[ 0,0,0],[ 0,0,0],[ 0,0,0],[0,0,0]
                ]
            ]
        )

        first_batch = next(iter(dataloader))

        assert first_batch.equal(expected)


class TestMLMDataLoader:
    def test_shapes(self, cdr3t_dataset):
        dataloader = dataloaders.MLMDataLoader(
            dataset=cdr3t_dataset,
            batch_size=3
        )

        masked, target = next(iter(dataloader))

        assert type(masked) == type(target) == torch.Tensor
        assert masked.dim() == 3
        assert target.dim() == 2
        assert masked.size(0) == target.size(0) == 3
        assert masked.size(1) == target.size(1) == 12
        assert masked.size(2) == 3


class TestAutoContrastiveDataLoader:
    def test_shapes(self, cdr3t_auto_contrastive_dataset):
        dataloader = dataloaders.AutoContrastiveDataLoader(
            dataset=cdr3t_auto_contrastive_dataset,
            batch_size=3
        )

        x, x_prime, masked, target = next(iter(dataloader))

        assert type(x) == type(x_prime) == type(masked) == type(target)\
            == torch.Tensor
        assert x.dim() ==  x_prime.dim() == masked.dim() == 3
        assert target.dim() == 2
        assert x.size(0) == x_prime.size(0)\
            == masked.size(0) == target.size(0) == 3
        assert x.size(1) == masked.size(1) == target.size(1) == 12
        assert x_prime.size(1) in (6, 7, 12)
        assert x.size(2) == x_prime.size(2) == masked.size(2) == 3


class TestEpitopeContrastiveDataLoader:
    def test_init(self, cdr3t_epitope_contrastive_dataset):
        dataloader = dataloaders.EpitopeContrastiveDataLoader(
            dataset=cdr3t_epitope_contrastive_dataset,
            num_workers=3
        )

        assert dataloader.dataset == cdr3t_epitope_contrastive_dataset
        assert dataloader.batch_size == 2
        assert type(dataloader.sampler) == SequentialSampler
        assert dataloader.sampler.data_source == cdr3t_epitope_contrastive_dataset
        assert type(dataloader.batch_sampler) == BatchSampler
        assert dataloader.batch_sampler.sampler == dataloader.sampler
        assert dataloader.batch_sampler.batch_size == 2
        assert dataloader.num_workers == 3