'''
pretrain.py
purpose: Main executable python script which trains a cdr3bert instance and
         saves checkpoint models and training logs.
author: Yuta Nagano
ver: 2.0.5
'''


import argparse
from hyperparams import hyperparams
import os
import pandas as pd
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
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
        '-g', '--gpus',
        default=0,
        type=int,
        help='The number of GPUs to utilise. If set to 0, the training ' + \
            'loop will be run on the CPU.'
    )
    parser.add_argument(
        '-q', '--no-progressbars',
        action='store_true',
        help='Running with this flag will suppress the output of any ' + \
            'progress bars. This may be useful to keep the output stream ' + \
            'clean when running the program on the cluster, especially if ' + \
            'the program will be run in distributed training mode ' + \
            '(accross multiple GPUs).'
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

    # Create a path to a target directory corresponding to the specified run_id
    tr_dir = os.path.join('training_runs',run_id)

    # If there already exists a directory at the specified path/name
    if os.path.isdir(tr_dir):
        # If overwrite=True, delete the preexisting directory along with all
        # contents, to free up that path address.
        if overwrite: shutil.rmtree(tr_dir)
        # Otherwise, keep modifying the target directory path in a systematic
        # way until we find a directory path that does not yet exist.
        else:
            suffix_int = 1
            new_tr_dir = f'{tr_dir}_{suffix_int}'
            while os.path.isdir(new_tr_dir):
                suffix_int += 1
                new_tr_dir = f'{tr_dir}_{suffix_int}'
            # Quick user feedback
            print(
                f'A directory {tr_dir} already exists. Target directory now '\
                f'modified to {new_tr_dir}.'
            )
            tr_dir = new_tr_dir

    # Create the directory
    os.mkdir(tr_dir)

    # Return the path to that directory
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


def set_env_vars(master_addr: str, master_port: str) -> None:
    '''
    Set some environment variables that are required when spawning parallel
    processes using torch.nn.parallel.DistributedDataParallel.
    '''
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port


def print_with_deviceid(msg: str, device: torch.device) -> None:
    print(f'[{device}]: {msg}')


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
        no_progressbars: bool,
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
    for x, y in tqdm(dataloader, desc=f'[{device}]', disable=no_progressbars):
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
        no_progressbars: bool,
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
    for x, y in tqdm(dataloader, desc=f'[{device}]', disable=no_progressbars):
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
    device,
    hyperparameters: dict,
    save_dir_path: str,
    no_progressbars: bool = False,
    multiprocess: bool = False,
    world_size: int = 1
) -> None:
    '''
    Train an instance of a CDR3BERT model using unlabelled CDR3 data. If
    multiprocess=True, perform necessary setup for synchronised parallel
    processing and splitting of the dataloader. Once training is finished, save
    the trained model as well as a log of training stats in the directory
    specified.
    '''
    # Initialise process group if multiprocessing
    if multiprocess:
        dist.init_process_group(
            backend='nccl',
            rank=device,
            world_size=world_size
        )

    # Wrap device identifier with torch.device
    device = torch.device(device)

    # Instantiate model, dataloader and any other objects required for training
    print_with_deviceid('Instantiating cdr3bert model...', device)

    model = Cdr3Bert(
        num_encoder_layers=hyperparameters['num_encoder_layers'],
        d_model=hyperparameters['d_model'],
        nhead=hyperparameters['nhead'],
        dim_feedforward=hyperparameters['dim_feedforward']
    ).to(device)

    # Wrap the model with DistributedDataParallel if multiprocessing
    if multiprocess:
        model = DistributedDataParallel(model, device_ids=[device])

    print_with_deviceid('Loading cdr3 data into memory...', device)

    train_dataset = CDR3Dataset(path_to_csv=hyperparameters['path_train_data'])
    # Create a split dataloader if multiprocessing
    # NOTE: batch_optimisation is currently unsupported in multiprocessing, as
    #       specifying distributed_sampler is mutually exclusive with having
    #       batch_optimisation = True.
    # TODO: implement randomised seeding for pseudorandom number generator at
    #       runtime within main().
    if multiprocess:
        train_sampler = DistributedSampler(
            dataset=train_dataset,
            num_replicas=world_size,
            rank=device.index,
            shuffle=True,
            seed=0
        )
        train_dataloader = CDR3DataLoader(
            dataset=train_dataset,
            batch_size=hyperparameters['batch_size'],
            distributed_sampler=train_sampler
        )
    # Otherwise, create a standard dataloader
    else:
        train_dataloader = CDR3DataLoader(
            dataset=train_dataset,
            batch_size=hyperparameters['batch_size'],
            shuffle=True,
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

    print_with_deviceid(
        'Instantiating other misc. objects for training...',
        device
    )

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
    print_with_deviceid('Commencing training...', device)

    # Create dictionaries to keep a log of training stats
    stats_log = dict()

    # Take note of the starting time
    start_time = time.time()

    # Begin training loop
    for epoch in range(1, hyperparameters['num_epochs']+1):
        # Do an epoch through the training data
        print_with_deviceid(f'Beginning epoch {epoch}...', device)
        train_stats = train_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimiser,
            no_progressbars,
            device
        )

        # Validate model performance
        print_with_deviceid('Validating model...', device)
        valid_stats = validate(
            model,
            val_dataloader,
            loss_fn,
            no_progressbars,
            device
        )

        # Quick feedback
        print_with_deviceid(
            f'training loss: {train_stats["train_loss"]:.3f} | '\
            f'validation loss: {valid_stats["valid_loss"]:.3f}',
            device
        )
        print_with_deviceid(
            f'training accuracy: {train_stats["train_acc"]:.3f} | '\
            f'validation accuracy: {valid_stats["valid_acc"]:.3f}',
            device
        )

        # Log stats
        stats_log[epoch] = {**train_stats, **valid_stats}

    print_with_deviceid('Training finished.', device)

    # Evaluate the model on jumbled validation data to ensure that the model is
    # learning something more than just amino acid residue frequencies.
    print_with_deviceid(
        'Evaluating model on jumbled validation data...',
        device
    )
    val_dataloader.jumble = True
    jumbled_valid_stats = validate(
        model,
        val_dataloader,
        loss_fn,
        no_progressbars,
        device
    )
    
    # Quick feedback
    print_with_deviceid(
        f'jumbled loss: {jumbled_valid_stats["jumble_loss"]:.3f} | '\
        f'jumbled accuracy: {jumbled_valid_stats["jumble_acc"]:.3f}',
        device
    )

    # Save the results of the jumbled data validation in the log.
    stats_log[hyperparameters['num_epochs']+1] = jumbled_valid_stats

    # Print the total time taken to train to the console.
    time_taken = int(time.time() - start_time)
    print_with_deviceid(
        f'Total time taken: {time_taken}s ({time_taken / 60} min)',
        device
    )

    # Save log as csv
    print_with_deviceid(
        f'Saving training log to '\
        f'{os.path.join(save_dir_path, f"train_stats_{device}.csv")}...',
        device
    )
    log_df = pd.DataFrame.from_dict(data=stats_log, orient='index')
    log_df.to_csv(
        os.path.join(save_dir_path, f'train_stats_{device}.csv'),
        index_label='epoch'
    )

    # Save trained model (if multiprocessing, this step is only done by process
    # with rank 0, i.e. the process on GPU 0)
    if not multiprocess or (multiprocess and device.index == 0):
        print_with_deviceid(
            f'Saving model to '\
            f'{os.path.join(save_dir_path, "trained_model.ptnn")}...',
            device
        )
        torch.save(
            model.cpu(),
            os.path.join(save_dir_path, 'trained_model.ptnn')
        )
    
    # If multiprocessing, then clean up by terminating the process group
    if multiprocess:
        dist.destroy_process_group()


def main(
    run_id: str,
    n_gpus: int = 0,
    test_mode: bool = False,
    no_progressbars: bool = False,
) -> None:
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
    # the test mode preset, along with setting the run ID to 'test'. Otherwise,
    # set the hyperparameters to what is contained in the hyperparameters file.
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

    # If multiple GPUs are expected:
    if n_gpus > 1:
        print(
            f'{n_gpus} CUDA devices expected, setting up distributed '\
            'training...'
        )
        # Set the required environment variables to properly create a process
        # group
        set_env_vars(master_addr='localhost', master_port='7777')
        # Spawn parallel processes each running train() on a different GPU
        mp.spawn(
            train,
            args=(hp, dirpath, no_progressbars, True, n_gpus),
            nprocs=n_gpus
        )
    # If there is one GPU available:
    elif n_gpus == 1:
        print('1 CUDA device expected, running training loop on cuda device...')
        train(0, hp, dirpath, no_progressbars)
    # If there are no GPUs available:
    else:
        print('No CUDA devices expected, running training loop on cpu...')
        train('cpu', hp, dirpath, no_progressbars)


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_command_line_arguments()
    
    main(args.run_id, args.gpus, args.test, args.no_progressbars)