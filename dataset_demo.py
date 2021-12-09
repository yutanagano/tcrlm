import torch
import matplotlib.pyplot as plt
from data_handling import CDR3Dataset

val_set = CDR3Dataset('data/val.csv',padding=32)

figure = plt.figure(figsize=(8,8))
cols, rows = 2, 3

for i in range(rows):
    sample_idx = torch.randint(len(val_set), size=(1,)).item()
    atchley, one_hot = val_set[sample_idx]

    figure.add_subplot(rows, cols, cols*i+1)
    plt.title(val_set.get_cdr3(sample_idx) + ' (atchley)')
    plt.axis('off')
    plt.imshow(atchley,vmin=-1,vmax=1)

    figure.add_subplot(rows, cols, cols*i+2)
    plt.title(val_set.get_cdr3(sample_idx) + ' (one hot)')
    plt.axis('off')
    plt.imshow(one_hot)

plt.show()