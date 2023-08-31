from tqdm import tqdm

from src.nn.training_delegate import TrainingDelegate
from src.config_reader.training_object_collection import TrainingObjectCollection
from src.nn import performance_measure


class MlmTrainingDelegate(TrainingDelegate):
    def run_training_epoch_and_return_metrics_for(
        self, training_object_collection: TrainingObjectCollection
    ) -> dict:
        model = training_object_collection.model
        dataloader = training_object_collection.training_dataloader
        loss_fn = training_object_collection.loss_functions["cross_entropy_loss"]
        optimiser = training_object_collection.optimiser
        device = training_object_collection.device

        current_process_not_on_first_gpu = device.index != 0

        model.train()

        total_loss = 0
        total_lr = 0
        divisor = 0

        for masked_tcrs, mlm_targets in tqdm(
            dataloader, disable=current_process_not_on_first_gpu
        ):
            num_samples = len(masked_tcrs)

            masked_tcrs = masked_tcrs.to(device)
            mlm_targets = mlm_targets.to(device)
            logits = model(masked_tcrs)

            optimiser.zero_grad()
            loss = loss_fn(logits.flatten(0, 1), mlm_targets.view(-1))
            loss.backward()
            optimiser.step()

            total_loss += loss.item() * num_samples
            total_lr += optimiser.lr * num_samples
            divisor += num_samples

        return {"loss": total_loss / divisor, "lr": total_lr / divisor}

    def validate_and_return_metrics_for(
        self, training_object_collection: TrainingObjectCollection
    ) -> dict:
        model = training_object_collection.model
        dataloader = training_object_collection.validation_dataloader
        loss_fn = training_object_collection.loss_functions["cross_entropy_loss"]
        device = training_object_collection.device

        current_process_not_on_first_gpu = device.index != 0

        model.eval()

        total_loss = 0
        total_acc = 0
        total_top5_acc = 0
        divisor = 0

        for masked_tcrs, mlm_targets in tqdm(
            dataloader, disable=current_process_not_on_first_gpu
        ):
            num_samples = len(masked_tcrs)

            masked_tcrs = masked_tcrs.to(device)
            mlm_targets = mlm_targets.to(device)

            logits = model(masked_tcrs)

            loss = loss_fn(logits.flatten(0, 1), mlm_targets.view(-1))

            total_loss += loss.item() * num_samples
            total_acc += performance_measure.mlm_acc(logits, mlm_targets) * num_samples
            total_top5_acc += performance_measure.mlm_topk_acc(logits, mlm_targets, 5) * num_samples
            divisor += num_samples

        return {
            "valid_loss": total_loss / divisor,
            "valid_acc": total_acc / divisor,
            "valid_top5_acc": total_top5_acc / divisor,
        }
