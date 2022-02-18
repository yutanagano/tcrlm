'''
pretrain.py
purpose: Main executable python script which trains a cdr3bert instance and
         saves checkpoint models and training logs.
author: Yuta Nagano
ver: 1.6.0
'''


import argparse
from hyperparams import hyperparams
import os
import pandas as pd
import time
import torch
import torch.distributed as dist
from tqdm import tqdm
import shutil

from source.cdr3bert import Cdr3Bert
from source.data_handling import CDR3Dataset, CDR3DataLoader
from source.training import create_padding_mask, AdamWithScheduling


# Hyperparameter preset for testing mode
hyperparams_test = {
    'path_train_data': 'tests/data/mock_data.csv',
    'path_valid_data': 'tests/data/mock_data.csv',
    'num_encoder_layers': 16,
    'd_model': 16,
    'nhead': 4,
    'dim_feedforward': 128,
    'batch_size': 512,
    'batch_optimisation': True,
    'lr_scheduling': True,
    'lr': 0.001,
    'optim_warmup': 5,
    'num_epochs': 20,
}


# Helper functions for training
def parse_command_line_arguments() -> argparse.Namespace:
    # Parse command line arguments using argparse
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description='Main training loop script for CDR3 BERT pre-training.'
    )

    # Add relevant arguments
    parser.add_argument(
        '-t', '--test',
        action='store_true',
        help='Run the training script in testing mode. Used for debugging. ' + \
            'Note that when using this flag, the run_id of the training ' + \
            'run will always be set to "test" regardless of what is ' + \
            'specified in the command line argument. If a "test" training ' + \
            'run directory already exists, this will be deleted along with ' + \
            'any contents.'
    )
    parser.add_argument(
        'run_id',
        help='Give this particular training run a unique ID.'
    )

    # Parse arguments read from sys.argv and return the resulting NameSpace
    # object containing the argument data
    return parser.parse_args()


def create_training_run_directory(run_id: str, overwrite: bool = False) -> str:
    '''
    If the 'training_runs' parent directory does not yet exist in the working
    directory, create it, and then create a training run directory inside it
    corresponding to the supplied run_id string. Return a string path to the
    newly created training run directory. If 'overwrite' is specified, then
    if a directory corresponding to the specified run_id already exists, delete
    it and overwrite it. Otherwise, this existing files with the same name will
    raise an exception.
    '''
    # Create parent directory if not already existent
    if not os.path.isdir('training_runs'): os.mkdir('training_runs')

    # Create new directory corresponding to the specified run_id
    tr_dir = os.path.join('training_runs',run_id)

    # If overwrite=True, then check for pre-existing directory with the same
    # name as the specified run_id and delete it along with its contents
    if overwrite and os.path.isdir(tr_dir):
        shutil.rmtree(tr_dir)

    os.mkdir(tr_dir)

    return tr_dir


def write_hyperparameters(hyperparameters: dict, dirpath: str) -> None:
    '''
    Write hyperparameters to a text file named 'hyperparams.txt' in the
    specified directory.
    '''
    print(
        f'Writing hyperparameters to '\
        f'{os.path.join(dirpath, "hyperparams.txt")}...'
    )
    with open(os.path.join(dirpath, 'hyperparams.txt'), 'w') as f:
        f.writelines([f'{k}: {hyperparameters[k]}\n' for k in hyperparameters])


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
        criterion: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        device: torch.device
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
        x = x.to(device)
        y = y.to(device)

        # Create padding mask for batch
        padding_mask = create_padding_mask(x)

        # Forward pass
        logits = model(x=x, padding_mask=padding_mask)
        logits = logits.view(-1,logits.size(-1))
        y = y.view(-1)

        # Backward pass
        optimiser.zero_grad()

        loss = criterion(logits,y)
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
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: torch.device
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
        x = x.to(device)
        y = y.to(device)

        # Create padding mask for batch
        padding_mask = create_padding_mask(x)

        # Forward pass
        logits = model(x=x, padding_mask=padding_mask)
        logits = logits.view(-1,logits.size(-1))
        y = y.view(-1)

        # Loss calculation
        loss = criterion(logits,y)

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


def train(
    hyperparameters: dict,
    device: torch.device,
    save_dir_path: str
) -> None:
    '''
    Train an instance of a CDR3BERT model using unlabelled CDR3 data. Save the
    trained model as well as a log of training stats in the directory specified.
    '''
    # Instantiate model, dataloader and any other objects required for training
    print('Instantiating cdr3bert model...')

    model = Cdr3Bert(
        num_encoder_layers=hyperparameters['num_encoder_layers'],
        d_model=hyperparameters['d_model'],
        nhead=hyperparameters['nhead'],
        dim_feedforward=hyperparameters['dim_feedforward']
    )

    if torch.cuda.device_count() > 1:
        print(
            f'Detected {torch.cuda.device_count()} gpus, '\
            'setting up distributed training...'
        )
        model = torch.nn.DataParallel(model)
    model.to(device)

    print('Loading cdr3 data into memory...')

    train_dataset = CDR3Dataset(path_to_csv=hyperparameters['path_train_data'])
    train_dataloader = CDR3DataLoader(
        dataset=train_dataset,
        batch_size=hyperparameters['batch_size'],
        batch_optimisation=hyperparameters['batch_optimisation']
    )

    val_dataset = CDR3Dataset(
        path_to_csv=hyperparameters['path_valid_data'],
        p_mask_random=0,
        p_mask_keep=0
    )
    val_dataloader = CDR3DataLoader(
        dataset=val_dataset,
        batch_size=hyperparameters['batch_size']
    )

    print('Instantiating other misc. objects for training...')

    optimiser = AdamWithScheduling(
        params=model.parameters(),
        lr=hyperparameters['lr'],
        betas=(0.9,0.999),
        eps=1e-08,
        lr_multiplier=1,
        d_model=hyperparameters['d_model'],
        n_warmup_steps=hyperparameters['optim_warmup'],
        scheduling=hyperparameters['lr_scheduling']
    )
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=21,label_smoothing=0.1)

    # Train model for a set number of epochs
    print('Commencing training.')

    # Create dictionaries to keep a log of training stats
    stats_log = dict()

    # Take note of the starting time
    start_time = time.time()

    # Begin training loop
    for epoch in range(1, hyperparameters['num_epochs']+1):
        # Do an epoch through the training data
        print(f'Beginning epoch {epoch}...')
        train_stats = train_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimiser,
            device
        )

        # Validate model performance
        print('Validating model...')
        valid_stats = validate(
            model,
            val_dataloader,
            loss_fn,
            device
        )

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

    # Evaluate the model on jumbled validation data to ensure that the model is
    # learning something more than just amino acid residue frequencies.
    print('Evaluating model on jumbled validation data...')
    val_dataloader.jumble = True
    jumbled_valid_stats = validate(
        model,
        val_dataloader,
        loss_fn,
        device
    )
    
    # Quick feedback
    print(
        f'jumbled loss: {jumbled_valid_stats["jumble_loss"]:.3f} | '\
        f'jumbled accuracy: {jumbled_valid_stats["jumble_acc"]:.3f}'
    )

    # Save the results of the jumbled data validation in the log.
    stats_log[hyperparameters['num_epochs']+1] = jumbled_valid_stats

    # Print the total time taken to train to the console.
    time_taken = int(time.time() - start_time)
    print(f'Total time taken: {time_taken}s ({time_taken / 60} min)')

    # Save log as csv
    print(
        f'Saving training log to '\
        f'{os.path.join(save_dir_path, "train_stats.csv")}...'
    )
    log_df = pd.DataFrame.from_dict(data=stats_log, orient='index')
    log_df.to_csv(
        os.path.join(save_dir_path, 'train_stats.csv'),
        index_label='epoch'
    )

    # Save trained model
    print(
        f'Saving model to '\
        f'{os.path.join(save_dir_path, "trained_model.ptnn")}...'
    )
    torch.save(model.cpu(), os.path.join(save_dir_path, 'trained_model.ptnn'))


def main(run_id: str, test_mode: bool = False) -> None:
    '''
    Main execution.

    Args:
    run_id:     A string which acts as a unique identifier of this training run.
                Used to name the directory in which the results from this run
                will be stored.
    test_mode:  If true, the program will run using a set of hyperparameters
                meant specifically for testing (e.g. use toy data, etc.).
    '''
    # If the program is being run in testing mode, set the hyperparameters to
    # the test mode preset. Otherwise, set the hyperparameters to what is
    # contained in the hyperparameters file.
    if test_mode:
        hp = hyperparams_test
        run_id = 'test'
    else:
        hp = hyperparams

    # Claim space to store results of training run by creating a new directory
    # based on the training id specified above
    dirpath = create_training_run_directory(run_id, overwrite=test_mode)

    # Save a text file containing info of current run's hyperparameters
    write_hyperparameters(hp, dirpath)

    # Detect training device(s)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Training will commence on device: {DEVICE}.')
    
    # Train a model instance
    train(hp, DEVICE, dirpath)


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_command_line_arguments()
    
    main(args.run_id, args.test)