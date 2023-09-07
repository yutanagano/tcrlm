from tqdm import tqdm

from src.nn.training_delegate import TrainingDelegate
from src.config_reader.training_object_collection import TrainingObjectCollection
from src.nn import performance_measure


class ClTrainingDelegate(TrainingDelegate):
    def run_training_epoch_and_return_metrics_for(
        self, training_object_collection: TrainingObjectCollection
    ) -> dict:
        model = training_object_collection.model
        dataloader = training_object_collection.training_dataloader
        cross_entropy_loss_fn = training_object_collection.loss_functions[
            "cross_entropy_loss"
        ]
        contrastive_loss_fn = training_object_collection.loss_functions[
            "contrastive_loss"
        ]
        optimiser = training_object_collection.optimiser
        device = training_object_collection.device

        current_process_not_on_first_gpu = device.index != 0

        model.train()

        total_loss = 0
        total_lr = 0
        divisor = 0

        for double_view_batch, double_view_positives_mask, masked_tcrs, mlm_targets in tqdm(
            dataloader, disable=current_process_not_on_first_gpu
        ):
            num_samples = len(double_view_batch)

            double_view_batch = double_view_batch.to(device)
            double_view_positives_mask = double_view_positives_mask.to(device)
            masked_tcrs = masked_tcrs.to(device)
            mlm_targets = mlm_targets.to(device)

            double_view_batch_embeddings, mlm_logits = model(
                double_view_batch, masked_tcrs
            )

            optimiser.zero_grad()
            loss = contrastive_loss_fn(
                double_view_batch_embeddings, double_view_positives_mask
            ) + cross_entropy_loss_fn(mlm_logits.flatten(0, 1), mlm_targets.view(-1))
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
        cross_entropy_loss_fn = training_object_collection.loss_functions[
            "cross_entropy_loss"
        ]
        contrastive_loss_fn = training_object_collection.loss_functions[
            "contrastive_loss"
        ]
        device = training_object_collection.device

        current_process_not_on_first_gpu = device.index != 0

        model.eval()

        total_cont_loss = 0
        total_positive_distance = 0
        total_negative_distance = 0
        total_mlm_loss = 0
        total_mlm_acc = 0
        divisor = 0

        for double_view_batch, double_view_positives_mask, masked_tcrs, mlm_targets in tqdm(
            dataloader, disable=current_process_not_on_first_gpu
        ):
            num_samples = len(double_view_batch)

            double_view_batch = double_view_batch.to(device)
            double_view_positives_mask = double_view_positives_mask.to(device)
            masked_tcrs = masked_tcrs.to(device)
            mlm_targets = mlm_targets.to(device)

            double_view_batch_embeddings, mlm_logits = model(double_view_batch, masked_tcrs)

            contrastive_loss = contrastive_loss_fn(
                double_view_batch_embeddings, double_view_positives_mask
            )
            mlm_loss = cross_entropy_loss_fn(
                mlm_logits.flatten(0, 1), mlm_targets.view(-1)
            )

            total_cont_loss += contrastive_loss.item() * num_samples
            total_positive_distance += performance_measure.average_positive_distance(double_view_batch_embeddings, double_view_positives_mask) * num_samples
            total_negative_distance += performance_measure.average_negative_distance(double_view_batch_embeddings, double_view_positives_mask) * num_samples
            total_mlm_loss += mlm_loss.item() * num_samples
            total_mlm_acc += performance_measure.mlm_acc(mlm_logits, mlm_targets) * num_samples
            divisor += num_samples

        return {
            "valid_cont_loss": total_cont_loss / divisor,
            "valid_positive_distance": total_positive_distance / divisor,
            "valid_negative_distance": total_negative_distance / divisor,
            "valid_mlm_loss": total_mlm_loss / divisor,
            "valid_mlm_acc": total_mlm_acc / divisor,
        }
