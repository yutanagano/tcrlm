from .. import models
from ..datahandling import tokenisers
from ..datahandling.dataloaders import MLMDataLoader
from ..datahandling.datasets import TCRDataset
from ..metrics import AdjustedCELoss, mlm_acc, mlm_topk_acc
from ..models.wrappers import MLMModelWrapper
from ..optim import AdamWithScheduling
from .training_pipeline import TrainingPipeline
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


class MLMPipeline(TrainingPipeline):
    @staticmethod
    def training_obj_factory(config: dict, rank: int) -> tuple:
        # Instantiate model
        model = getattr(models, config["model"]["class"])(**config["model"]["config"])
        model.to(rank)
        model = DDP(MLMModelWrapper(model), device_ids=[rank])

        # Load train/valid data
        tokeniser = getattr(tokenisers, config["data"]["tokeniser"]["class"])(
            **config["data"]["tokeniser"]["config"]
        )
        train_ds = TCRDataset(data=config["data"]["train_path"], tokeniser=tokeniser)
        valid_ds = TCRDataset(data=config["data"]["valid_path"], tokeniser=tokeniser)
        train_dl = MLMDataLoader(
            dataset=train_ds,
            sampler=DistributedSampler(train_ds),
            **config["data"]["dataloader"]["config"],
        )
        valid_dl = MLMDataLoader(
            dataset=valid_ds,
            p_mask_random=0,
            p_mask_keep=0,
            **config["data"]["dataloader"]["config"],
        )

        # Loss functions
        loss_fn = AdjustedCELoss(label_smoothing=0.1)

        # Optimiser
        optimiser = AdamWithScheduling(
            params=model.parameters(),
            d_model=config["model"]["config"]["d_model"],
            **config["optim"]["optimiser"]["config"],
        )

        return model, train_dl, valid_dl, (loss_fn,), optimiser

    @staticmethod
    def train_func(
        model: DDP, dl: MLMDataLoader, loss_fns: tuple, optimiser, rank: int
    ) -> dict:
        loss_fn = loss_fns[0]

        model.train()

        total_loss = 0
        total_lr = 0
        divisor = 0

        for x, y in tqdm(dl, disable=rank):
            num_samples = len(x)

            x = x.to(rank)
            y = y.to(rank)
            logits = model(x)

            optimiser.zero_grad()
            loss = loss_fn(logits.flatten(0, 1), y.view(-1))
            loss.backward()
            optimiser.step()

            total_loss += loss.item() * num_samples
            total_lr += optimiser.lr * num_samples
            divisor += num_samples

        return {"loss": total_loss / divisor, "lr": total_lr / divisor}

    @staticmethod
    @torch.no_grad()
    def valid_func(model: DDP, dl: MLMDataLoader, loss_fns: tuple, rank: int) -> dict:
        loss_fn = loss_fns[0]

        model.eval()

        total_loss = 0
        total_acc = 0
        total_top5_acc = 0
        divisor = 0

        for x, y in tqdm(dl, disable=rank):
            num_samples = len(x)

            x = x.to(rank)
            y = y.to(rank)

            logits = model(x)

            loss = loss_fn(logits.flatten(0, 1), y.view(-1))

            total_loss += loss.item() * num_samples
            total_acc += mlm_acc(logits, y) * num_samples
            total_top5_acc += mlm_topk_acc(logits, y, 5) * num_samples
            divisor += num_samples

        return {
            "valid_loss": total_loss / divisor,
            "valid_acc": total_acc / divisor,
            "valid_top5_acc": total_top5_acc / divisor,
        }
