from tqdm import tqdm

from src.model_trainer.training_delegate import TrainingDelegate
from src.model_trainer.training_object_collection import TrainingObjectCollection
from src import metric


class AclTrainingDelegate(TrainingDelegate):
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

        for anchor_tcrs, positive_pair_tcrs, masked_tcrs, mlm_targets in tqdm(
            dataloader, disable=current_process_not_on_first_gpu
        ):
            num_samples = len(anchor_tcrs)

            anchor_tcrs = anchor_tcrs.to(device)
            positive_pair_tcrs = positive_pair_tcrs.to(device)
            masked_tcrs = masked_tcrs.to(device)
            mlm_targets = mlm_targets.to(device)

            anchor_tcr_embeddings, positive_pair_tcr_embeddings, mlm_logits = model(
                anchor_tcrs, positive_pair_tcrs, masked_tcrs
            )

            optimiser.zero_grad()
            loss = contrastive_loss_fn(
                anchor_tcr_embeddings, positive_pair_tcr_embeddings
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

        total_cont_loss = 0
        total_mlm_loss = 0
        total_aln = 0
        total_unf = 0
        total_mlm_acc = 0
        divisor = 0

        for anchor_tcrs, positive_pair_tcrs, masked_tcrs, mlm_targets in tqdm(
            dataloader, disable=current_process_not_on_first_gpu
        ):
            num_samples = len(anchor_tcrs)

            anchor_tcrs = anchor_tcrs.to(device)
            positive_pair_tcrs = positive_pair_tcrs.to(device)
            masked_tcrs = masked_tcrs.to(device)
            mlm_targets = mlm_targets.to(device)

            model.train()  # turn dropout on for contrastive eval, as it adds noise
            anchor_tcr_embeddings = model.module.bert.get_vector_representations_of(
                anchor_tcrs
            )
            positive_pair_tcr_embeddings = (
                model.module.bert.get_vector_representations_of(positive_pair_tcrs)
            )

            model.eval()
            mlm_logits = model.module.bert.get_mlm_token_predictions_for(masked_tcrs)

            contrastive_loss = contrastive_loss_fn(
                anchor_tcr_embeddings, positive_pair_tcr_embeddings
            )
            mlm_loss = cross_entropy_loss_fn(
                mlm_logits.flatten(0, 1), mlm_targets.view(-1)
            )

            total_cont_loss += contrastive_loss.item() * num_samples
            total_mlm_loss += mlm_loss.item() * num_samples
            total_aln += (
                metric.alignment_paired(
                    anchor_tcr_embeddings, positive_pair_tcr_embeddings
                ).item()
                * num_samples
            )
            total_unf += metric.uniformity(anchor_tcr_embeddings).item() * num_samples
            total_mlm_acc += metric.mlm_acc(mlm_logits, mlm_targets) * num_samples
            divisor += num_samples

        return {
            "valid_cont_loss": total_cont_loss / divisor,
            "valid_mlm_loss": total_mlm_loss / divisor,
            "valid_aln": total_aln / divisor,
            "valid_unf": total_unf / divisor,
            "valid_mlm_acc": total_mlm_acc / divisor,
        }
