'''
Main executable python script which performs pretraining of a Cdr3Bert model
instance on unlabelled CDR3 data.
'''


import argparse
from pathlib import Path
from source.datahandling.dataloaders import Cdr3PretrainDataLoader
from source.datahandling.datasets import Cdr3PretrainDataset
from source.nn.grad import AdamWithScheduling
import source.nn.metrics as metrics
from source.nn.models import Cdr3Bert, Cdr3BertPretrainWrapper
import source.utils.fileio as fileio
import source.utils.misc as misc
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import CrossEntropyLoss, Module
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from typing import Union


# Helper functions for training
def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Main training loop script for CDR3 BERT pre-training.'
    )

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
            'is specified, this feature is disabled and the batch size '
            'will be kept constant regardless of the number of CUDA devices.'
    )
    parser.add_argument(
        '-q', '--no-progressbars',
        action='store_true',
        help='Running with this flag will suppress the output of any '
            'progress bars. This may be useful to keep the output stream '
            'clean when running the program on the cluster, especially '
            'if the program will be run in distributed training mode '
            '(accross multiple GPUs).'
    )
    parser.add_argument(
        '-t', '--test',
        action='store_true',
        help='Run the training script in testing mode. Used for '
            'debugging. Note that when using this flag, the run_id of the '
            'training run will always be set to "test" regardless of what '
            'is specified in the command line argument. The '
            'hyperparameters path will also always be set to '
            '"tests/data/pretrain_hyperparams.csv". If a "test" '
            'pretraining run directory already exists, this will be '
            'deleted along with any contents.'
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

    return parser.parse_args()


def instantiate_model(
    hyperparameters: dict,
    device: torch.device
) -> Cdr3BertPretrainWrapper:
    'Instantiate a Cdr3Bert model and wrap it in a pretraining wrapper.'
    
    bert = Cdr3Bert(
        num_encoder_layers=hyperparameters['num_encoder_layers'],
        d_model=hyperparameters['d_model'],
        nhead=hyperparameters['nhead'],
        dim_feedforward=hyperparameters['dim_feedforward'],
        activation=hyperparameters['activation']
    )
    return Cdr3BertPretrainWrapper(bert).to(device)


def train_epoch(
    model: Cdr3BertPretrainWrapper,
    dataloader: Cdr3PretrainDataLoader,
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
    total_top5_acc = 0

    total_acc_third0 = []
    total_top5_acc_third0 = []

    total_acc_third1 = []
    total_top5_acc_third1 = []

    total_acc_third2 = []
    total_top5_acc_third2 = []

    total_lr = 0

    start_time = time.time()

    # Iterate through the dataloader
    for x, y in tqdm(dataloader, desc=f'[{device}]', disable=no_progressbars):
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        logits = model(x)

        # Backward pass
        optimiser.zero_grad()

        loss = criterion(
            logits.view(-1,20),
            y.view(-1)
        )
        loss.backward()

        optimiser.step()
        
        # Increment stats
        total_loss += loss.item()

        total_acc += metrics.pretrain_accuracy(logits,y)
        total_top5_acc += metrics.pretrain_topk_accuracy(logits,y,5)

        total_acc_third0.append(metrics.pretrain_accuracy_third(logits,x,y,0))
        total_top5_acc_third0.append(metrics.pretrain_topk_accuracy_third(
                                                            logits,x,y,5,0))

        total_acc_third1.append(metrics.pretrain_accuracy_third(logits,x,y,1))
        total_top5_acc_third1.append(metrics.pretrain_topk_accuracy_third(
                                                            logits,x,y,5,1))

        total_acc_third2.append(metrics.pretrain_accuracy_third(logits,x,y,2))
        total_top5_acc_third2.append(metrics.pretrain_topk_accuracy_third(
                                                            logits,x,y,5,2))

        total_lr += optimiser.lr

    elapsed = time.time() - start_time

    # Return a dictionary with stats averaged to represent per-sample values.
    # Since the loss value at each batch is averaged over the samples in it,
    # the accumulated loss/accuracy values should be divided by the number of
    # batches in the dataloader. The exception here are the accuracy values
    # calculated per CDR3 segment (thirds), as these metrics are not always
    # available for every batch. These are filtered and averaged as available.
    divisor = len(dataloader)

    return {
        'train_loss'            : total_loss / divisor,
        'train_acc'             : total_acc / divisor,
        'train_top5_acc'        : total_top5_acc / divisor,
        'train_acc_third0'      : misc.dynamic_fmean(total_acc_third0),
        'train_top5_acc_third0' : misc.dynamic_fmean(total_top5_acc_third0),
        'train_acc_third1'      : misc.dynamic_fmean(total_acc_third1),
        'train_top5_acc_third1' : misc.dynamic_fmean(total_top5_acc_third1),
        'train_acc_third2'      : misc.dynamic_fmean(total_acc_third2),
        'train_top5_acc_third2' : misc.dynamic_fmean(total_top5_acc_third2),
        'avg_lr'                : total_lr / divisor,
        'epoch_time'            : elapsed
    }


@torch.no_grad()
def validate(
    model: Cdr3BertPretrainWrapper,
    dataloader: Cdr3PretrainDataLoader,
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
    total_top5_acc = 0

    total_acc_third0 = []
    total_top5_acc_third0 = []

    total_acc_third1 = []
    total_top5_acc_third1 = []

    total_acc_third2 = []
    total_top5_acc_third2 = []

    # Iterate through the dataloader
    for x, y in tqdm(dataloader, desc=f'[{device}]', disable=no_progressbars):
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        logits = model(x)

        # Loss calculation
        loss = criterion(
            logits.view(-1,20),
            y.view(-1)
        )

        # Increment stats
        total_loss += loss.item()

        total_acc += metrics.pretrain_accuracy(logits,y)
        total_top5_acc += metrics.pretrain_topk_accuracy(logits,y,5)

        total_acc_third0.append(metrics.pretrain_accuracy_third(logits,x,y,0))
        total_top5_acc_third0.append(metrics.pretrain_topk_accuracy_third(
                                                            logits,x,y,5,0))

        total_acc_third1.append(metrics.pretrain_accuracy_third(logits,x,y,1))
        total_top5_acc_third1.append(metrics.pretrain_topk_accuracy_third(
                                                            logits,x,y,5,1))

        total_acc_third2.append(metrics.pretrain_accuracy_third(logits,x,y,2))
        total_top5_acc_third2.append(metrics.pretrain_topk_accuracy_third(
                                                            logits,x,y,5,2))

    # Decide on appropriate name for the statistic calculated based on the
    # dataloader's jumble status
    if dataloader.dataset.jumble:
        stat_prefix = 'jumble'
    else:
        stat_prefix = 'valid'
    
    # Return a dictionary with stats
    divisor = len(dataloader)

    return {
        f'{stat_prefix}_loss'               : total_loss / divisor,
        f'{stat_prefix}_acc'                : total_acc / divisor,
        f'{stat_prefix}_top5_acc'           : total_top5_acc / divisor,
        f'{stat_prefix}_acc_third0'         : misc.dynamic_fmean(
                                                        total_acc_third0),
        f'{stat_prefix}_top5_acc_third0'    : misc.dynamic_fmean(
                                                        total_top5_acc_third0),
        f'{stat_prefix}_acc_third1'         : misc.dynamic_fmean(
                                                        total_acc_third1),
        f'{stat_prefix}_top5_acc_third1'    : misc.dynamic_fmean(
                                                        total_top5_acc_third1),
        f'{stat_prefix}_acc_third2'         : misc.dynamic_fmean(
                                                        total_acc_third2),
        f'{stat_prefix}_top5_acc_third2'    : misc.dynamic_fmean(
                                                        total_top5_acc_third2),
    }


def train(
    device,
    hyperparameters: dict,
    training_run_dir: str,
    no_progressbars: bool = False,
    world_size: int = 1,
    test_mode: bool = False
) -> None:
    '''
    Pretrain an instance of a dr3Bert model using unlabelled CDR3 data. If
    world_size > 1 (distributed training), perform necessary setup for
    synchronised parallel processing and splitting of the dataloader. Once
    training is finished, save the trained model as well as a log of training
    stats in the directory specified.
    '''

    # If world_size is > 1, we are running in distributed mode
    distributed = (world_size > 1)

    # Initialise process group if distributed
    if distributed:
        dist.init_process_group(
            backend='nccl',
            rank=device,
            world_size=world_size
        )

    # Wrap device identifier with torch.device
    device = torch.device(device)

    # Initialise record manager
    record_manager = fileio.TrainingRecordManager(
        training_run_dir=training_run_dir,
        distributed=distributed,
        device=device,
        test_mode=test_mode
    )

    # Instantiate model
    misc.print_with_deviceid('Instantiating cdr3bert model...', device)
    model = instantiate_model(hyperparameters, device)

    # Wrap the model with DistributedDataParallel if distributed
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device])

    # Load training and validation data
    misc.print_with_deviceid('Loading cdr3 data into memory...', device)

    # NOTE: batch_optimisation is currently unsupported in distributed mode, as
    #       specifying distributed_sampler is mutually exclusive with having
    #       batch_optimisation = True.
    if distributed and hyperparameters['batch_optimisation']:
        # Automatically disable batch_optimisation
        hyperparameters['batch_optimisation'] = False
        # And inform user
        if device.index == 0:
            print(
                'WARNING: batch_optimisation has been set in hyperparameters, '
                'but this setting is currently unsupported when running in '
                'distributed training mode.'
            )

    # Training data
    train_dataset = Cdr3PretrainDataset(
        data=hyperparameters['path_train_data']
    )
    train_dataloader = Cdr3PretrainDataLoader(
            dataset=train_dataset,
            batch_size=hyperparameters['train_batch_size'],
            shuffle=True,
            num_workers=4,
            distributed=distributed,
            batch_optimisation=hyperparameters['batch_optimisation'],
            num_replicas=world_size,
            rank=device.index
    )

    # Validation data
    val_dataset = Cdr3PretrainDataset(
        data=hyperparameters['path_valid_data'],
        p_mask_random=0,
        p_mask_keep=0
    )
    val_dataloader = Cdr3PretrainDataLoader(
        dataset=val_dataset,
        batch_size=hyperparameters['valid_batch_size'],
        num_workers=4,
        batch_optimisation=True
    )

    # Instantiate loss function and optimiser
    misc.print_with_deviceid(
        'Instantiating other misc. objects for training...',
        device
    )
    loss_fn = CrossEntropyLoss(ignore_index=21,label_smoothing=0.1)
    optimiser = AdamWithScheduling(
        params=model.parameters(),
        d_model=hyperparameters['d_model'],
        n_warmup_steps=hyperparameters['optim_warmup'],
        lr=hyperparameters['lr'],
        decay=hyperparameters['lr_decay']
    )

    misc.print_with_deviceid('Commencing training...', device)

    stats_log = dict()
    start_time = time.time()

    # Main training loop
    for epoch in range(1, hyperparameters['num_epochs']+1):
        misc.print_with_deviceid(f'Beginning epoch {epoch}...', device)

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
        misc.print_with_deviceid('Validating model...', device)
        valid_stats = validate(
            model,
            val_dataloader,
            loss_fn,
            device,
            no_progressbars
        )

        # Quick feedback
        misc.print_with_deviceid(
            f'training loss: {train_stats["train_loss"]:.3f} | '\
            f'training accuracy: {train_stats["train_acc"]:.3f}',
            device
        )
        misc.print_with_deviceid(
            f'validation loss: {valid_stats["valid_loss"]:.3f} | '\
            f'validation accuracy: {valid_stats["valid_acc"]:.3f}',
            device
        )

        # Log stats
        stats_log[epoch] = {**train_stats, **valid_stats}

    misc.print_with_deviceid('Training finished.', device)

    # Evaluate the model on jumbled validation data to ensure that the model is
    # learning something more than just amino acid residue frequencies.
    misc.print_with_deviceid(
        'Evaluating model on jumbled validation data...',
        device
    )
    val_dataloader.dataset.jumble = True
    jumbled_valid_stats = validate(
        model,
        val_dataloader,
        loss_fn,
        device,
        no_progressbars
    )
    
    # Quick feedback
    misc.print_with_deviceid(
        f'jumbled loss: {jumbled_valid_stats["jumble_loss"]:.3f} | '\
        f'jumbled accuracy: {jumbled_valid_stats["jumble_acc"]:.3f}',
        device
    )

    # Save the results of the jumbled data validation in the log.
    stats_log[hyperparameters['num_epochs']+1] = jumbled_valid_stats

    time_taken = int(time.time() - start_time)
    misc.print_with_deviceid(
        f'Total time taken: {time_taken}s ({time_taken / 60} min)',
        device
    )

    # Save results
    record_manager.save_log(log_dict=stats_log)
    record_manager.save_model(model=model, name='pretrained')
    
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

    training_run_dir = fileio.create_training_run_directory(
        working_directory=working_directory,
        run_id=run_id,
        mode='pretrain',
        overwrite=test_mode
    )
    
    hyperparams = fileio.parse_hyperparams(hyperparams_path)

    fileio.write_hyperparameters(
        hyperparameters=hyperparams,
        training_run_dir=training_run_dir
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
        misc.set_env_vars(master_addr='localhost', master_port='7777')
        mp.spawn(
            train,
            args=(
                hyperparams,
                training_run_dir,
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
            no_progressbars=no_progressbars,
            test_mode=test_mode
        )

    else:
        print('No CUDA devices expected, running training loop on cpu...')
        train(
            device='cpu',
            hyperparameters=hyperparams,
            training_run_dir=training_run_dir,
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