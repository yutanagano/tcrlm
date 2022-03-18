'''
finetune.py
purpose: Main executable python script which performs finetuning of a Cdr3Bert
         model instance on labelled CDR3 data.
author: Yuta Nagano
ver: 1.0.1
'''


import argparse
from hyperparams import finetune_hyperparams
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import CrossEntropyLoss, Module
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from source.cdr3bert import Cdr3Bert, Cdr3BertFineTuneWrapper
from source.data_handling import Cdr3FineTuneDataset, Cdr3FineTuneDataLoader
from source.training import create_training_run_directory, \
    print_with_deviceid, set_env_vars, write_hyperparameters, save_log, \
    save_model, AdamWithScheduling


# Hyperparameter preset for testing mode
hyperparams_test = {
    'pretrain_id': 'test',
    'path_train_data': os.path.join(
        'tests', 'data', 'mock_labelled_data.csv'),
    'path_valid_data': os.path.join(
        'tests', 'data', 'mock_labelled_data.csv'),
    'train_batch_size': 6,
    'valid_batch_size': 6,
    'lr_scheduling': True,
    'lr': 0.001,
    'optim_warmup': 5,
    'num_epochs': 3,
}


# Helper functions
def parse_command_line_arguments() -> argparse.Namespace:
    '''
    Parse command line arguments using argparse.
    '''
    parser = argparse.ArgumentParser(
        description='Main training loop script for CDR3 BERT fine-tuning.'
    )

    # Add relevant arguments
    parser.add_argument(
        '-g', '--gpus',
        default=0,
        type=int,
        help='The number of GPUs to utilise. If set to 0, the training ' + \
            'loop will be run on the CPU.'
    )
    parser.add_argument(
        '-b','--fixed-batch-size',
        action='store_true',
        help='Without this option, when the program is running in ' + \
            'distributed training mode, the batch size will be adaptively ' + \
            'modified based on how many CUDA devices are available. That ' + \
            'is, new_batch_size = old_batch_size // nGPUs. If this flag ' + \
            'is specified, this feature is disabled and the per-GPU batch ' + \
            'size will be kept constant regardless of CUDA device numbers.'
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
        '-t', '--test',
        action='store_true',
        help='Run the training script in testing mode. Used for debugging. ' + \
            'Note that when using this flag, the run_id of the training ' + \
            'run will always be set to "test" regardless of what is ' + \
            'specified in the command line argument. If a "test" training ' + \
            'run directory already exists, this will be deleted along with ' + \
            'any contents and replaced with the results of the new test run.'
    )
    parser.add_argument(
        'run_id',
        help='Give this particular pretrain run a unique ID.'
    )

    # Parse arguments read from sys.argv and return the resulting NameSpace
    # object containing the argument data
    return parser.parse_args()


def load_pretrained_model(
    hyperparameters: dict,
    device: torch.device
) -> Cdr3BertFineTuneWrapper:
    '''
    Load a pretrained Cdr3Bert model from a specified pretrain run.
    '''
    # Load pretrained model
    saved_model_location = os.path.join(
        'pretrain_runs', hyperparameters['pretrain_id'], 'pretrained.ptnn'
    )
    bert = torch.load(saved_model_location).bert
    
    return Cdr3BertFineTuneWrapper(bert).to(device)


@torch.no_grad()
def accuracy(x: torch.Tensor, y: torch.Tensor) -> float:
    '''
    Calculate the accuracy of model predictions.
    '''
    correct = torch.argmax(x,dim=1) == y
    return (correct.sum() / correct.size(0)).item()


def train_epoch(
    model: Module,
    dataloader: Cdr3FineTuneDataLoader,
    criterion: Module,
    optimiser,
    device: torch.device,
    no_progressbars: bool = False
) -> dict:
    '''
    Train the given model through one epoch of data from the given dataloader.
    '''
    model.train()

    total_loss = 0
    total_acc = 0
    total_lr = 0

    start_time = time.time()

    # Iterate through the dataloader
    for x_a, x_b, y in \
        tqdm(dataloader, desc=f'[{device}]', disable=no_progressbars):
        x_a = x_a.to(device)
        x_b = x_b.to(device)
        y = y.to(device)

        # Forward pass
        logits = model(x_a, x_b)

        # Backward pass
        optimiser.zero_grad()

        loss = criterion(logits, y)
        loss.backward()

        optimiser.step()

        # Increment stats
        total_loss += loss.item()
        total_acc += accuracy(logits,y)
        total_lr += optimiser.lr

    elapsed = time.time() - start_time

    # Return a dictionary with stats averaged to represent per-sample values.
    # Since the loss value at each batch is averaged over the samples in it,
    # the accumulated loss/accuracy/lr values should be divided by the number of
    # batches in the dataloader.
    return {
        'train_loss': total_loss / len(dataloader),
        'train_acc' : total_acc / len(dataloader),
        'avg_lr'    : total_lr / len(dataloader),
        'epoch_time': elapsed
    }


@torch.no_grad()
def validate(
    model: Cdr3BertFineTuneWrapper,
    dataloader: Cdr3FineTuneDataLoader,
    criterion: Module,
    device: torch.device,
    no_progressbars: bool = False
) -> dict:
    '''
    Validates the given model's performance by calculating loss and other stats
    from the data in the given dataloader.
    '''
    model.eval()
    
    total_loss = 0
    total_acc = 0

    # Iterate through the dataloader
    for x_a, x_b, y in \
        tqdm(dataloader, desc=f'[{device}]', disable=no_progressbars):
        x_a = x_a.to(device)
        x_b = x_b.to(device)
        y = y.to(device)

        # Forward pass
        logits = model(x_a, x_b)

        # Loss calculation
        loss = criterion(logits,y)

        # Increment stats
        total_loss += loss.item()
        total_acc += accuracy(logits,y)
    
    # Return a dictionary with stats averaged to represent per-sample values.
    # Since the loss value at each batch is averaged over the samples in it,
    # the accumulated loss/accuracy/lr values should be divided by the number of
    # batches in the dataloader.
    return {
        'valid_loss': total_loss / len(dataloader),
        'valid_acc' : total_acc / len(dataloader)
    }


def train(
    device,
    hyperparameters: dict,
    save_dir_path: str,
    no_progressbars: bool = False,
    world_size: int = 1,
    test_mode: bool = False
) -> None:
    '''
    Fine-tune an instance of a Cdr3Bert model using labelled CDR3 data. If
    world_size > 1 (distributed training), perform necessary setup for
    synchronised parallel processing and splitting of the dataloader. Once
    training is finished, save the trained model as well as a log of training
    stats in the directory specified.
    '''
    # If world_size is > 1, we are running in distributed mode
    distributed = (world_size > 1)

    # Initialise process group if running in distributed mode
    if distributed:
        dist.init_process_group(
            backend='nccl',
            rank=device,
            world_size=world_size
        )
    
    # For easier use, from here wrap the device identifier with torch.device
    device = torch.device(device)

    # Load model
    print_with_deviceid(
        'Loading pretrained model from pretrain run ID: '
        f'{hyperparameters["pretrain_id"]}...',
        device
    )
    model = load_pretrained_model(hyperparameters, device)

    # Take note of d_model, used later when instantiating optimiser
    d_model = model.bert.d_model

    # Wrap the model with DistributedDataParallel if distributed
    if distributed: model = DistributedDataParallel(model, device_ids=[device])

    # Load training and validation data
    print_with_deviceid('Loading cdr3 data into memory...', device)

    # Training data
    train_dataset = Cdr3FineTuneDataset(
        path_to_csv = hyperparameters['path_train_data']
    )
    
    if distributed:
        # NOTE: the set_epoch() method on the distributed sampler will need to
        #       be called at the start of every epoch in order for the shuffling
        #       to be done correctly at every epoch.
        distributed_sampler = DistributedSampler(
            dataset=train_dataset,
            num_replicas=world_size,
            rank=device.index,
            shuffle=True,
            seed=0
        )
        train_dataloader = Cdr3FineTuneDataLoader(
            dataset=train_dataset,
            batch_size=hyperparameters['train_batch_size'],
            num_workers=4,
            distributed_sampler=distributed_sampler
        )
    else:
        train_dataloader = Cdr3FineTuneDataLoader(
            dataset=train_dataset,
            batch_size=hyperparameters['train_batch_size'],
            num_workers=4,
            shuffle=True
        )

    # Validation data
    val_dataset = Cdr3FineTuneDataset(
        path_to_csv=hyperparameters['path_valid_data'],
    )
    val_dataloader = Cdr3FineTuneDataLoader(
        dataset=val_dataset,
        batch_size=hyperparameters['valid_batch_size'],
        num_workers=4
    )

    # Instantiate loss function and optimiser
    print_with_deviceid(
        'Instantiating other misc. objects for training...',
        device
    )
    loss_fn = CrossEntropyLoss()
    optimiser = AdamWithScheduling(
        params=model.parameters(),
        d_model=d_model,
        n_warmup_steps=hyperparameters['optim_warmup'],
        lr=hyperparameters['lr'],
        decay=False
    )

    print_with_deviceid('Commencing training...', device)

    stats_log = dict()
    start_time = time.time()

    # Main training loop
    for epoch in range(1, hyperparameters['num_epochs']+1):
        print_with_deviceid(f'Beginning epoch {epoch}...', device)

        # If in distributed mode, inform the distributed sampler that a new
        # epoch is beginning
        if distributed: distributed_sampler.set_epoch(epoch)

        # Do an epoch through the training data
        train_stats = train_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimiser,
            device,
            no_progressbars
        )

        # Validate model performance
        print_with_deviceid('Validating model...', device)
        valid_stats = validate(
            model,
            val_dataloader,
            loss_fn,
            device,
            no_progressbars
        )

        # Quick feedback
        print_with_deviceid(
            f'training loss: {train_stats["train_loss"]:.3f} | '\
            f'training accuracy: {train_stats["train_acc"]:.3f}',
            device
        )
        print_with_deviceid(
            f'validation loss: {valid_stats["valid_loss"]:.3f} | '\
            f'validation accuracy: {valid_stats["valid_acc"]:.3f}',
            device
        )
        
        # Log stats
        stats_log[epoch] = {**train_stats, **valid_stats}

    print_with_deviceid('Training finished.', device)

    time_taken = int(time.time() - start_time)
    print_with_deviceid(
        f'Total time taken: {time_taken}s ({time_taken / 60} min)',
        device
    )

    # Save results
    save_log(stats_log, save_dir_path, distributed, device)
    save_model(
        model,
        'finetuned',
        save_dir_path,
        distributed,
        device,
        test_mode
    )

    # If distributed, then clean up by terminating the process group
    if distributed: dist.destroy_process_group()


def main(
    run_id: str,
    n_gpus: int = 0,
    fixed_batch_size: bool = False,
    no_progressbars: bool = False,
    test_mode: bool = False
) -> None:
    '''
    Main execution.

    Args:
    run_id:             A string which acts as a unique identifier of this
                        training run. Used to name the directory in which the
                        results from this run will be stored.
    n_gpus              An integer value which signifies how many CUDA-capable
                        devices are expected to be available.
    fixed_batch_size    Disables adaptive batch_size modification in distributed
                        training mode.
    no_progressbars     Whether to suppress progressbar outputs to the output
                        stream or not.
    test_mode:          If true, the program will run using a set of
                        hyperparameters meant specifically for testing (e.g. use
                        toy data, etc.).
    '''
    # If the program is being run in testing mode, set the hyperparameters to
    # the testing preset, along with setting the run ID to 'test'. Otherwise,
    # set the hyperparameters to what is contained in the hyperparameters file.
    if test_mode:
        hp = hyperparams_test
        run_id = 'test'
    else:
        hp = finetune_hyperparams

    # Claim space to store results of training run by creating a new directory
    # based on the training ID specified above.
    dirpath = create_training_run_directory(
        run_id,
        mode='finetune',
        overwrite=test_mode
    )

    # Save a text file containing info of current run's hyperparameters
    write_hyperparameters(hp, dirpath)

    # If multiple GPUs are expected:
    if n_gpus > 1:
        print(
            f'{n_gpus} CUDA devices expected, setting up distributed '\
            'training...'
        )

        # To help with debugging in case of funky device allocation (especially
        # on the cluster) print the number of CUDA devices detected at runtime
        print(
            f'{torch.cuda.device_count()} CUDA devices detected...'
        )

        # If not fixed_batch_size, modify the batch size based on how many CUDA
        # devices the training process will be split over
        if not fixed_batch_size: hp['train_batch_size'] //= n_gpus

        # Set the required environment variables to properly create a process
        # group
        set_env_vars(master_addr='localhost', master_port='7777')

        # Spawn parallel processes each running train() on a different GPU
        mp.spawn(
            train,
            args=(hp, dirpath, no_progressbars, n_gpus, test_mode),
            nprocs=n_gpus
        )

        # If in test mode, verify that the trained models saved from all
        # processes are equivalent (i.e. they all have the same weights).
        if test_mode: compare_models(dirpath, n_gpus)

    # If there is one GPU available:
    elif n_gpus == 1:
        print('1 CUDA device expected, running training loop on cuda device...')
        train(
            device=0,
            hyperparameters=hp,
            save_dir_path=dirpath,
            no_progressbars=no_progressbars,
            test_mode=test_mode
        )
    
    # If there are no GPUs available:
    else:
        print('No CUDA devices expected, running training loop on cpu...')
        train(
            device='cpu',
            hyperparameters=hp,
            save_dir_path=dirpath,
            no_progressbars=no_progressbars,
            test_mode=test_mode
        )


if __name__ == '__main__':
    args = parse_command_line_arguments()

    main(
        args.run_id,
        args.gpus,
        args.fixed_batch_size,
        args.no_progressbars,
        args.test
    )