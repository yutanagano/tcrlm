import torch
from torch import BoolTensor, FloatTensor


@torch.no_grad()
def average_positive_distance(
    tcr_representations: FloatTensor, positives_mask: BoolTensor
) -> float:
    all_distances = torch.cdist(tcr_representations, tcr_representations, p=2)
    positive_distances = all_distances * positives_mask
    positive_distances_summed = positive_distances.sum().item()
    num_positive_distances_in_batch = positives_mask.sum().item()
    return positive_distances_summed / num_positive_distances_in_batch


@torch.no_grad()
def average_negative_distance(
    tcr_representations: FloatTensor, positives_mask: BoolTensor
) -> float:
    all_distances = torch.cdist(tcr_representations, tcr_representations, p=2)
    identity_mask = torch.eye(
        len(tcr_representations), device=positives_mask.device
    ).logical_not()
    negatives_mask = torch.logical_and(positives_mask.logical_not(), identity_mask)
    negative_distances = all_distances * negatives_mask
    negative_distances_summed = negative_distances.sum().item()
    num_negative_distances_in_batch = negatives_mask.sum().item()
    return negative_distances_summed / num_negative_distances_in_batch
