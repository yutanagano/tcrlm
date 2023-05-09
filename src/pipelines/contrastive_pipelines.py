from .. import models
from .. import metrics
from ..datahandling import tokenisers
from ..datahandling.dataloaders import ContrastiveDataLoader
from ..datahandling.datasets import AutoContrastiveDataset
from ..models.wrappers import CLModelWrapper
from ..metrics import AdjustedCELoss, alignment_paired, mlm_acc, uniformity
from ..optim import AdamWithScheduling
from .training_pipeline import TrainingPipeline
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


class CLPipeline(TrainingPipeline):
    @staticmethod
    def train_func(
        model: DDP, dl: ContrastiveDataLoader, loss_fns: tuple, optimiser, rank: int
    ) -> dict:
        mlm_loss_fn, cont_loss_fn = loss_fns

        model.train()

        total_loss = 0
        total_lr = 0
        divisor = 0

        for x, x_prime, masked, target in tqdm(dl, disable=rank):
            num_samples = len(x)

            x = x.to(rank)
            x_prime = x_prime.to(rank)
            masked = masked.to(rank)
            target = target.to(rank)

            z, z_prime, mlm_logits = model(x, x_prime, masked)

            optimiser.zero_grad()
            loss = cont_loss_fn(z, z_prime) + mlm_loss_fn(
                mlm_logits.flatten(0, 1), target.view(-1)
            )
            loss.backward()
            optimiser.step()

            total_loss += loss.item() * num_samples
            total_lr += optimiser.lr * num_samples
            divisor += num_samples

        return {"loss": total_loss / divisor, "lr": total_lr / divisor}

    @staticmethod
    @torch.no_grad()
    def valid_func(
        model: DDP, dl: ContrastiveDataLoader, loss_fns: tuple, rank: int
    ) -> dict:
        mlm_loss_fn, cont_loss_fn = loss_fns

        total_cont_loss = 0
        total_mlm_loss = 0
        total_aln = 0
        total_unf = 0
        total_mlm_acc = 0
        divisor = 0

        for x, x_prime, masked, target in tqdm(dl, disable=rank):
            num_samples = len(x)

            x = x.to(rank)
            x_prime = x_prime.to(rank)
            masked = masked.to(rank)
            target = target.to(rank)

            model.train()  # turn dropout on for contrastive eval, as it adds noise
            z = model.module.embedder.embed(x)
            z_prime = model.module.embedder.embed(x_prime)

            model.eval()
            mlm_logits = model.module.embedder.mlm(masked)

            cont_loss = cont_loss_fn(z, z_prime)
            mlm_loss = mlm_loss_fn(mlm_logits.flatten(0, 1), target.view(-1))

            total_cont_loss += cont_loss.item() * num_samples
            total_mlm_loss += mlm_loss.item() * num_samples
            total_aln += alignment_paired(z, z_prime).item() * num_samples
            total_unf += uniformity(z).item() * num_samples
            total_mlm_acc += mlm_acc(mlm_logits, target) * num_samples
            divisor += num_samples

        return {
            "valid_cont_loss": total_cont_loss / divisor,
            "valid_mlm_loss": total_mlm_loss / divisor,
            "valid_aln": total_aln / divisor,
            "valid_unf": total_unf / divisor,
            "valid_mlm_acc": total_mlm_acc / divisor,
        }
    


class ACLPipeline(CLPipeline):
    @staticmethod
    def training_obj_factory(config: dict, rank: int) -> tuple:
        # Instantiate model
        model = getattr(models, config["model"]["class"])(**config["model"]["config"])
        model.load_state_dict(torch.load(config["model"]["pretrain_state_dict_path"]))
        model.to(rank)
        model = DDP(CLModelWrapper(model), device_ids=[rank])

        # Load train/valid data
        tokeniser = getattr(tokenisers, config["data"]["tokeniser"]["class"])(
            **config["data"]["tokeniser"]["config"]
        )
        train_ds = AutoContrastiveDataset(
            data=config["data"]["train_path"],
            tokeniser=tokeniser,
            **config["data"]["dataset"]["config"],
        )
        valid_ds = AutoContrastiveDataset(
            data=config["data"]["valid_path"],
            tokeniser=tokeniser,
            censoring_lhs=False,
            censoring_rhs=False,
        )
        train_dl = ContrastiveDataLoader(
            dataset=train_ds,
            sampler=DistributedSampler(train_ds),
            **config["data"]["dataloader"]["config"],
        )
        valid_dl = ContrastiveDataLoader(
            dataset=valid_ds,
            p_mask_random=0,
            p_mask_keep=0,
            **config["data"]["dataloader"]["config"],
        )

        # Loss functions
        mlm_loss_fn = AdjustedCELoss(label_smoothing=0.1)
        cont_loss_fn = getattr(metrics, config["optim"]["contrastive_loss"]["class"])(
            **config["optim"]["contrastive_loss"]["config"]
        )

        # Optimiser
        optimiser = AdamWithScheduling(
            params=model.parameters(),
            d_model=config["model"]["config"]["d_model"],
            **config["optim"]["optimiser"]["config"],
        )

        return model, train_dl, valid_dl, (mlm_loss_fn, cont_loss_fn), optimiser