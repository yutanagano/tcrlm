from src.datahandling import dataloaders
import torch
from torch.utils.data import BatchSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler


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


    def test_init_distributed(self, cdr3t_dataset):
        dataloader = dataloaders.TCRDataLoader(
            dataset=cdr3t_dataset,
            batch_size=3,
            num_workers=3,
            distributed=True,
            num_replicas=2,
            rank=0
        )

        assert dataloader.dataset == cdr3t_dataset
        assert dataloader.batch_size == 3
        assert type(dataloader.sampler) == DistributedSampler
        assert dataloader.sampler.dataset == cdr3t_dataset
        assert dataloader.sampler.num_replicas == 2
        assert dataloader.sampler.rank == 0
        assert dataloader.sampler.shuffle == True
        assert dataloader.sampler.seed == 0
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
