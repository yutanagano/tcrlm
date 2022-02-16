'''
main.py
purpose: Main executable python script which trains a cdr3bert instance and
         saves checkpoint models and training logs.
author: Yuta Nagano
ver: 1.5.0
'''


import argparse
from hyperparams import hyperparams
import os
import pandas as pd
import time
import torch
from tqdm import tqdm
from source.cdr3bert import Cdr3Bert
from source.data_handling import CDR3Dataset, CDR3DataLoader
from source.training import create_padding_mask, AdamWithScheduling


# Detect training device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
GPU_COUNT = torch.cuda.device_count()


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


def train_epoch(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimiser: torch.optim.Optimizer
    ) -> dict:
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
    return {
        'train_loss': total_loss / len(dataloader),
        'train_acc' : total_acc / len(dataloader),
        'avg_lr' : total_lr / len(dataloader),
        'epoch_time': elapsed
    }


@torch.no_grad()
def validate(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader
    ) -> dict:
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

    # Decide on appropriate name for the statistic calculated based on the
    # dataloader's jumble status
    if dataloader.jumble:
        stat_names = ('jumble_loss','jumble_acc')
    else:
        stat_names = ('valid_loss','valid_acc')
    
    # Return a dictionary with stats
    return {
        stat_names[0]: total_loss / len(dataloader),
        stat_names[1]: total_acc / len(dataloader)
    }


if __name__ == '__main__':
    # Create argparser, add arguments
    parser = argparse.ArgumentParser(
        description='Main training loop script for CDR3 BERT pre-training.'
    )
    parser.add_argument(
        'run_id',
        help='Give this particular training run a unique ID.'
    )
    args = parser.parse_args()


    # Set run ID
    RUN_ID = args.run_id


    # Claim space to store results of training run by creating a new directory
    # based on the training id specified above
    dirpath = os.path.join('training_runs',RUN_ID)
    os.mkdir(dirpath)


    print(f'Training will commence on device: {DEVICE}.')


    # Instantiate model, dataloader and any other objects required for training
    print('Instantiating cdr3bert model...')

    model = Cdr3Bert(
        num_encoder_layers=hyperparams['num_encoder_layers'],
        d_model=hyperparams['d_model'],
        nhead=hyperparams['nhead'],
        dim_feedforward=hyperparams['dim_feedforward']
    )
    if GPU_COUNT > 1:
        print(f'Detected {GPU_COUNT} gpus, setting up distributed training...')
        model = torch.nn.DataParallel(model)
    model.to(DEVICE)

    print('Loading cdr3 data into memory...')

    train_dataset = CDR3Dataset(path_to_csv=hyperparams['path_train_data'])
    train_dataloader = CDR3DataLoader(
        dataset=train_dataset,
        batch_size=hyperparams['batch_size'],
        batch_optimisation=hyperparams['batch_optimisation']
    )

    val_dataset = CDR3Dataset(
        path_to_csv=hyperparams['path_valid_data'],
        p_mask_random=0,
        p_mask_keep=0
    )
    val_dataloader = CDR3DataLoader(
        dataset=val_dataset,
        batch_size=hyperparams['batch_size']
    )

    print('Instantiating other misc. objects for training...')

    optimiser = AdamWithScheduling(
        params=model.parameters(),
        lr=hyperparams['lr'],
        betas=(0.9,0.999),
        eps=1e-08,
        lr_multiplier=1,
        d_model=hyperparams['d_model'],
        n_warmup_steps=hyperparams['optim_warmup'],
        scheduling=hyperparams['lr_scheduling']
    )
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=21,label_smoothing=0.1)


    # Train model for a set number of epochs
    print('Commencing training.')

    # 1) Create dictionaries to keep a lof of training stats
    stats_log = dict()

    # 1) Take note of the starting time
    start_time = time.time()

    # 2) Begin training loop
    for epoch in range(1, hyperparams['num_epochs']+1):
        # Do an epoch through the training data
        print(f'Beginning epoch {epoch}...')
        train_stats = train_epoch(model,train_dataloader,optimiser)

        # Validate model performance
        print('Validating model...')
        valid_stats = validate(model,val_dataloader)

        # Quick feedback
        print(
            f'training loss: {train_stats["train_loss"]:.3f} | '\
            f'validation loss: {valid_stats["valid_loss"]:.3f}'
        )
        print(
            f'training accuracy: {train_stats["train_acc"]:.3f} | '\
            f'validation accuracy: {valid_stats["valid_acc"]:.3f}'
        )

        # Log stats
        stats_log[epoch] = {**train_stats, **valid_stats}

    print('Training finished.')

    print('Evaluating model on jumbled validation data...')
    val_dataloader.jumble = True
    jumbled_valid_stats = validate(model,val_dataloader)
    
    # Quick feedback
    print(
        f'jumbled loss: {jumbled_valid_stats["jumble_loss"]:.3f} | '\
        f'jumbled accuracy: {jumbled_valid_stats["jumble_acc"]:.3f}'
    )

    stats_log[hyperparams['num_epochs']+1] = jumbled_valid_stats

    time_taken = int(time.time() - start_time)
    print(f'Total time taken: {time_taken}s ({time_taken / 60} min)')


    # Save hyperparameters as csv
    print('Saving hyperparameters...')
    with open(os.path.join(dirpath, 'hyperparams.txt'), 'w') as f:
        f.writelines([f'{k}: {hyperparams[k]}\n' for k in hyperparams])

    # Save log as csv
    print('Saving training log...')
    log_df = pd.DataFrame.from_dict(data=stats_log, orient='index')
    log_df.to_csv(os.path.join(dirpath, 'train_stats.csv'),index_label='epoch')

    # Save trained model
    print('Saving model...')
    torch.save(model.cpu(), os.path.join(dirpath, 'trained_model.ptnn'))