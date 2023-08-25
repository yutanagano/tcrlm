from torch.utils.data import DataLoader

from src.data.batch_collator import BatchCollator


class TcrDataLoader(DataLoader):
    def __init__(self, *args, batch_collator: BatchCollator, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = batch_collator.collate_fn

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)
