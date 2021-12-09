import torch
from torch.utils.data import DataLoader
from data_handling import CDR3Dataset

val_set = CDR3Dataset('data/val.csv',padding=32)
val_loader = DataLoader(val_set, batch_size=64)

batch_features, batch_labels = next(iter(val_loader))

print(f"Feature batch shape: {batch_features.size()}")
print(f"Labels batch shape: {batch_labels.size()}")