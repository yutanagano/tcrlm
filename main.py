'''
main.py
purpose: Main executable python script which trains a cdr3bert instance and
         saves checkpoint models and training logs.
author: Yuta Nagano
ver: 1.1.0
'''


import pandas as pd
import time
import torch
from tqdm import tqdm
from source.cdr3bert import Cdr3Bert
from source.data_handling import CDR3Dataset, CDR3DataLoader
from source.training import create_padding_mask, ScheduledOptimiser


# Outline hyperparameters and settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Training will commence on device: {DEVICE}.')

BERT_NUM_ENCODER_LAYERS = 8
BERT_D_MODEL = 16
BERT_NHEAD = 4
BERT_DIM_FEEDFORWARD = 128

BATCH_SIZE = 512

OPTIM_WARMUP = 10_000

NUM_EPOCHS = 2


# Instantiate model, dataloader and any other objects required for training
print('Instantiating cdr3bert model...')

model = Cdr3Bert(num_encoder_layers=BERT_NUM_ENCODER_LAYERS,
                 d_model=BERT_D_MODEL,
                 nhead=BERT_NHEAD,
                 dim_feedforward=BERT_DIM_FEEDFORWARD)
model.to(DEVICE)

print('Loading cdr3 data into memory...')

train_dataset = CDR3Dataset(path_to_csv='data/train.csv')
train_dataloader = CDR3DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE)

val_dataset = CDR3Dataset(path_to_csv='data/val.csv')
val_dataloader = CDR3DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE)

print('Instantiating other misc. objects for training...')

optimiser = ScheduledOptimiser(optimiser=torch.optim.Adam(
                                            params=model.parameters(),
                                            lr=0,
                                            betas=(0.9, 0.98),
                                            eps=1e-9),
                               lr_multiplier=1,
                               d_model=BERT_D_MODEL,
                               n_warmup_steps=OPTIM_WARMUP)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=21,label_smoothing=0.1)


# Helper functions for training
@torch.no_grad()
def accuracy(x: torch.Tensor, y: torch.Tensor) -> float:
    '''
    Calculate the batch average of the accuracy of model predictions ignoring
    any padding tokens.
    '''
    mask = (y != 21)
    correct = torch.argmax(x,dim=-1) == y
    correct_masked = correct & mask
    return (correct_masked.sum() / mask.sum()).item()


def train_epoch(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                optimiser: torch.optim.Optimizer) -> dict:
    # Train the given model through one epoch of data from the given dataloader.
    # Ensure that the model is in training mode.
    model.train()
    # Initialise variables to keep track of stats throughout the epoch.
    total_loss = 0
    total_acc = 0
    total_lr = 0

    # Take note of the start time
    start_time = time.time()

    # Iterate through the dataloader
    for x, y in tqdm(dataloader):
        # Transfer batches to appropriate device
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Create padding mask for batch
        padding_mask = create_padding_mask(x)

        # Forward pass
        logits = model(x=x, padding_mask=padding_mask)
        logits = logits.view(-1,logits.size(-1))
        y = y.view(-1)

        # Backward pass
        optimiser.zero_grad()

        loss = loss_fn(logits,y)
        loss.backward()

        optimiser.step()
        
        # Increment stats
        total_loss += loss.item()
        total_acc += accuracy(logits,y)
        total_lr += optimiser.lr

    # Take note of elapsed time
    elapsed = time.time() - start_time

    # Return a dictionary with stats
    return {'train_loss': total_loss / len(dataloader),
            'train_acc' : total_acc / len(dataloader),
            'avg_lr' : total_lr / len(dataloader),
            'epoch_time': elapsed}


@torch.no_grad()
def validate(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader) -> dict:
    '''
    Validates the given model's performance by calculating loss and other stats
    from the data in the given dataloader.
    '''
    # Ensure that the model is in evaludation mode
    model.eval()
    # Initialise variables to keep track of stats over the minibatches.
    total_loss = 0
    total_acc = 0

    # Iterate through the dataloader
    for x, y in tqdm(dataloader):
        # Transfer batches to appropriate device
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Create padding mask for batch
        padding_mask = create_padding_mask(x)

        # Forward pass
        logits = model(x=x, padding_mask=padding_mask)
        logits = logits.view(-1,logits.size(-1))
        y = y.view(-1)

        # Loss calculation
        loss = loss_fn(logits,y)

        # Increment batch loss to total_loss
        total_loss += loss.item()
        total_acc += accuracy(logits,y)
    
    # Return a dictionary with stats
    return {'valid_loss': total_loss / len(dataloader),
            'valid_acc' : total_acc / len(dataloader)}


if __name__ == '__main__':
    # Train model for NUM_EPOCHS epochs
    print('Commencing training.')

    # 1) Create dictionaries to keep a lof of training stats
    stats_log = dict()

    # 1) Take note of the starting time
    start_time = time.time()

    # 2) Begin training loop
    for epoch in range(1, NUM_EPOCHS+1):
        print(f'Beginning epoch {epoch}...')
        # Do an epoch through the training data
        train_stats = train_epoch(model,train_dataloader,optimiser)
        print('Validating model...')
        # Validate model performance
        valid_stats = validate(model,val_dataloader)

        # Quick feedback
        print(f'training loss: {train_stats["train_loss"]:.3f} | '\
              f'validation loss: {valid_stats["valid_loss"]:.3f}')
        print(f'training accuracy: {train_stats["train_acc"]:.3f} | '\
              f'validation accuracy: {valid_stats["valid_acc"]:.3f}')

        # Log stats
        stats_log[epoch] = {**train_stats, **valid_stats}

    print('Training finished.')
    print(f'Total time taken: {int(time.time() - start_time)}s')

    # Convert log to dataframe
    log_df = pd.DataFrame.from_dict(data=stats_log,
                                    orient='index')

    # Save log as csv
    log_df.to_csv('train_stats.csv',index_label='epoch')