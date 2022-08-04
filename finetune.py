'''
Main executable python script which performs finetuning of a Cdr3Bert model
instance on labelled CDR3 data.
'''


import argparse
from pathlib import Path
from source.datahandling.dataloaders import Cdr3FineTuneDataLoader
from source.datahandling.datasets import Cdr3FineTuneDataset
from source.nn.grad import AdamWithScheduling
from source.nn.metrics import finetune_accuracy
from source.nn.models import TcrEmbedder, Cdr3BertFineTuneWrapper
import source.utils.fileio as fileio
from source.utils.misc import print_with_deviceid, set_env_vars
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import CrossEntropyLoss, Module
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from typing import Union


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
        help='The number of GPUs to utilise. If set to 0, the training '
            'loop will be run on the CPU.'
    )
    parser.add_argument(
        '-b','--fixed-batch-size',
        action='store_true',
        help='Without this option, when the program is running in '
            'distributed training mode, the batch size will be adaptively '
            'modified based on how many CUDA devices are available. That '
            'is, new_batch_size = old_batch_size // nGPUs. If this flag '
            'is specified, this feature is disabled and the per-GPU batch '
            'size will be kept constant regardless of CUDA device numbers.'
    )
    parser.add_argument(
        '-q', '--no-progressbars',
        action='store_true',
        help='Running with this flag will suppress the output of any '
            'progress bars. This may be useful to keep the output stream '
            'clean when running the program on the cluster, especially if '
            'the program will be run in distributed training mode '
            '(accross multiple GPUs).'
    )
    parser.add_argument(
        '-t', '--test',
        action='store_true',
        help='Run the training script in testing mode. Used for debugging. '
            'Note that when using this flag, the run_id of the training '
            'run will always be set to "test" regardless of what is '
            'specified in the command line argument. The hyperparameters '
            'path will also always be set to '
            '"tests/data/finetune_hyperparams.csv". If a "test" finetuning '
            'run directory already exists, this will be deleted along with '
            'any contents.'
    )
    parser.add_argument(
        'run_id',
        help='Give this particular pretrain run a unique ID.'
    )
    parser.add_argument(
        'hyperparams_path',
        help='Path to a csv file containing hyperparameter values to be '
            'used for this run.'
    )

    # Parse arguments read from sys.argv and return the resulting NameSpace
    # object containing the argument data
    return parser.parse_args()


def resolve_pretrained_model_paths(
    working_directory: Path,
    alpha_pretrain_id: str,
    beta_pretrain_id: str
) -> dict:
    pretrain_dir = working_directory / 'pretrain_runs'
    alpha_bert_path = pretrain_dir / alpha_pretrain_id / 'pretrained.ptnn'
    beta_bert_path = pretrain_dir / beta_pretrain_id / 'pretrained.ptnn'

    return {
        'alpha_bert_path': alpha_bert_path.resolve(),
        'beta_bert_path': beta_bert_path.resolve()
    }


def load_pretrained_model(
    pretrained_model_paths: dict,
    device: torch.device
) -> Cdr3BertFineTuneWrapper:
    alpha_bert = torch.load(pretrained_model_paths['alpha_bert_path']).bert
    beta_bert = torch.load(pretrained_model_paths['beta_bert_path']).bert
    embedder = TcrEmbedder(alpha_bert, beta_bert)
    
    return Cdr3BertFineTuneWrapper(embedder).to(device)


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

    # Ensure model is in training mode, but the dropout modules in BERT must
    # remain in evaluation mode, so execute custom trainmode setter.
    if type(model) == Cdr3BertFineTuneWrapper: model.custom_trainmode()
    else: model.module.custom_trainmode()

    total_loss = 0
    total_acc = 0
    total_lr = 0

    start_time = time.time()

    # Iterate through the dataloader
    for x_1a, x_1b, x_2a, x_2b, y in \
        tqdm(dataloader, desc=f'[{device}]', disable=no_progressbars):
        x_1a = x_1a.to(device)
        x_1b = x_1b.to(device)
        x_2a = x_2a.to(device)
        x_2b = x_2b.to(device)
        y = y.to(device)

        # Forward pass
        logits = model(x_1a, x_1b, x_2a, x_2b)

        # Backward pass
        optimiser.zero_grad()

        loss = criterion(logits, y)
        loss.backward()

        optimiser.step()

        # Increment stats
        total_loss += loss.item()
        total_acc += finetune_accuracy(logits,y)
        total_lr += optimiser.lr

    elapsed = time.time() - start_time

    # Return a dictionary with stats averaged to represent per-sample values.
    # Since the loss value at each batch is averaged over the samples in it,
    # the accumulated loss/accuracy/lr values should be divided by the number
    # of batches in the dataloader.
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
    for x_1a, x_1b, x_2a, x_2b, y in \
        tqdm(dataloader, desc=f'[{device}]', disable=no_progressbars):
        x_1a = x_1a.to(device)
        x_1b = x_1b.to(device)
        x_2a = x_2a.to(device)
        x_2b = x_2b.to(device)
        y = y.to(device)

        # Forward pass
        logits = model(x_1a, x_1b, x_2a, x_2b)

        # Loss calculation
        loss = criterion(logits,y)

        # Increment stats
        total_loss += loss.item()
        total_acc += finetune_accuracy(logits,y)
    
    # Return a dictionary with stats averaged to represent per-sample values.
    # Since the loss value at each batch is averaged over the samples in it,
    # the accumulated loss/accuracy/lr values should be divided by the number
    # of batches in the dataloader.
    return {
        'valid_loss': total_loss / len(dataloader),
        'valid_acc' : total_acc / len(dataloader)
    }


def train(
    device,
    hyperparameters: dict,
    training_run_dir: str,
    pretrained_model_paths: dict,
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

    # Initialise record manager
    record_manager = fileio.TrainingRecordManager(
        training_run_dir=training_run_dir,
        distributed=distributed,
        device=device,
        test_mode=test_mode
    )

    # Load model
    print_with_deviceid(
        'Loading pretrained model from pretrain run IDs: '
        f'alpha - {hyperparameters["alpha_pretrain_id"]}, '
        f'beta - {hyperparameters["beta_pretrain_id"]}...',
        device
    )
    model = load_pretrained_model(
        pretrained_model_paths=pretrained_model_paths,
        device=device
    )

    # Take note of d_model, used later when instantiating optimiser
    d_model = model.d_model

    # Wrap the model with DistributedDataParallel if distributed
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device])

    # Load training and validation data
    print_with_deviceid('Loading cdr3 data into memory...', device)

    # Training data
    train_dataset = Cdr3FineTuneDataset(
        data = hyperparameters['path_train_data']
    )
    train_dataloader = Cdr3FineTuneDataLoader(
        dataset=train_dataset,
        batch_size=hyperparameters['train_batch_size'],
        shuffle=True,
        num_workers=4,
        distributed=distributed,
        num_replicas=world_size,
        rank=device.index
    )

    # Validation data
    val_dataset = Cdr3FineTuneDataset(
        data=hyperparameters['path_valid_data'],
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
        decay=hyperparameters['lr_decay']
    )

    print_with_deviceid('Commencing training...', device)

    stats_log = dict()
    start_time = time.time()

    # Main training loop
    for epoch in range(1, hyperparameters['num_epochs']+1):
        print_with_deviceid(f'Beginning epoch {epoch}...', device)

        # If in distributed mode, inform the distributed sampler that a new
        # epoch is beginning
        if distributed: train_dataloader.sampler.set_epoch(epoch)

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
    record_manager.save_log(log_dict=stats_log)
    record_manager.save_model(model=model, name='finetuned')

    # If distributed, then clean up by terminating the process group
    if distributed:
        dist.destroy_process_group()


def main(
    working_directory: Union[Path, str],
    run_id: str,
    hyperparams_path: str,
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
    hyperparams_path    A path to a csv file containing hyperparameter values
                        to be used for this run.
    n_gpus              An integer value which signifies how many CUDA-capable
                        devices are expected to be available.
    fixed_batch_size    Disables adaptive batch_size modification in
                        distributed training mode.
    no_progressbars     Whether to suppress progressbar outputs to the output
                        stream or not.
    test_mode:          If true, the program will run using a set of
                        hyperparameters meant specifically for testing (e.g. 
                        use toy data, etc.).
    '''
    working_directory = fileio.resolved_path_from_maybe_str(working_directory)

    training_run_dir = fileio.create_training_run_directory(
        working_directory=working_directory,
        run_id=run_id,
        mode='finetune',
        overwrite=test_mode
    )
    
    hyperparams = fileio.parse_hyperparams(hyperparams_path)

    fileio.write_hyperparameters(
        hyperparameters=hyperparams,
        training_run_dir=training_run_dir
    )

    pretrained_model_paths = resolve_pretrained_model_paths(
        working_directory=working_directory,
        alpha_pretrain_id=hyperparams['alpha_pretrain_id'],
        beta_pretrain_id=hyperparams['beta_pretrain_id']
    )

    if n_gpus > 1:
        print(
            f'{n_gpus} CUDA devices expected, setting up distributed '
            'training...'
        )

        print(f'{torch.cuda.device_count()} CUDA devices detected...')

        # If not fixed_batch_size, modify the batch size based on how many CUDA
        # devices the training process will be split over
        if not fixed_batch_size:
            hyperparams['train_batch_size'] //= n_gpus

        # Set up a process group
        set_env_vars(master_addr='localhost', master_port='7777')
        mp.spawn(
            train,
            args=(
                hyperparams,
                training_run_dir,
                pretrained_model_paths,
                no_progressbars,
                n_gpus,
                test_mode
            ),
            nprocs=n_gpus
        )

    elif n_gpus == 1:
        print(
            '1 CUDA device expected, running training loop on cuda device...'
        )
        train(
            device=0,
            hyperparameters=hyperparams,
            training_run_dir=training_run_dir,
            pretrained_model_paths=pretrained_model_paths,
            no_progressbars=no_progressbars,
            test_mode=test_mode
        )
    
    else:
        print('No CUDA devices expected, running training loop on cpu...')
        train(
            device='cpu',
            hyperparameters=hyperparams,
            training_run_dir=training_run_dir,
            pretrained_model_paths=pretrained_model_paths,
            no_progressbars=no_progressbars,
            test_mode=test_mode
        )


if __name__ == '__main__':
    args = parse_command_line_arguments()
    main(
        working_directory=Path.cwd(),
        run_id=args.run_id,
        hyperparams_path=args.hyperparams_path,
        n_gpus=args.gpus,
        fixed_batch_size=args.fixed_batch_size,
        no_progressbars=args.no_progressbars,
        test_mode=args.test
    )