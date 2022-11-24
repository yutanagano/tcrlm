'''
An executable script to conduct simple contrastive learning on TCR models.
'''


import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from src import modules
from src.modules.embedder import MLMEmbedder
from src.datahandling import tokenisers
from src.datahandling.dataloaders import (
    UnsupervisedSimCLDataLoader,
    SupervisedSimCLDataLoader
)
from src.datahandling.datasets import (
    UnsupervisedSimCLDataset,
    SupervisedSimCLDataset
)
from src.metrics import AdjustedCELoss, SimCLoss, alignment_paired, uniformity
from src.utils import save
import torch
from torch import multiprocessing as mp
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Union


MODELS = {
    'SimCTE_CDR3BERT_cp': modules.SimCTE_CDR3BERT_cp
}

TOKENISERS = {
    'CDR3Tokeniser': tokenisers.CDR3Tokeniser
}
DATASETS = {
    'UnsupervisedSimCLDataset': UnsupervisedSimCLDataset,
    'SupervisedSimCLDataset': SupervisedSimCLDataset
}
DATALOADERS = {
    'UnsupervisedSimCLDataLoader': UnsupervisedSimCLDataLoader,
    'SupervisedSimCLDataLoader': SupervisedSimCLDataLoader
}


def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Simple contrastive learning training loop.'
    )
    parser.add_argument(
        '-d', '--working-directory',
        help='Path to tcr_embedder project working directory.'
    )
    parser.add_argument(
        '-n', '--name',
        help='Name of the training run.'
    )
    parser.add_argument(
        'config_path',
        help='Path to the training run config json file.'
    )
    return parser.parse_args()


def metric_feedback(metrics: dict) -> None:
    for metric in metrics:
        print(f'{metric}: {metrics[metric]}')


def train(
    model: MLMEmbedder,
    dl: DataLoader,
    simc_loss_fn,
    optimiser,
    device
) -> dict:
    model.train()

    total_loss = 0
    divisor = 0

    for x, x_prime, _, _ in tqdm(dl):
        num_samples = len(x)

        x = x.to(device)
        x_prime = x_prime.to(device)

        z = model.embed(x)
        z_prime = model.embed(x_prime)

        optimiser.zero_grad()
        loss = simc_loss_fn(z, z_prime)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * num_samples
        divisor += num_samples

    return {
        'loss': total_loss / divisor
    }


@torch.no_grad()
def validate(
    model: MLMEmbedder,
    dl: DataLoader,
    mlm_loss_fn,
    simc_loss_fn,
    device
) -> dict:
    if isinstance(dl, SupervisedSimCLDataLoader):
        model.eval()

    total_loss = 0
    total_mlm_loss = 0
    total_aln = 0
    total_unf = 0
    divisor = 0

    for x, x_prime, masked, target in tqdm(dl):
        num_samples = len(x)

        x = x.to(device)
        x_prime = x_prime.to(device)
        masked = masked.to(device)
        target = target.to(device)

        mlm_logits = model.mlm(masked)
        z = model.embed(x)
        z_prime = model.embed(x_prime)

        loss = simc_loss_fn(z, z_prime)
        mlm_loss = mlm_loss_fn(mlm_logits.flatten(0,1), target.view(-1))

        total_loss += loss.item() * num_samples
        total_mlm_loss += mlm_loss.item() * num_samples
        total_aln += alignment_paired(z, z_prime, alpha=2).item() * num_samples
        total_unf += uniformity(z, t=2).item() * num_samples
        divisor += num_samples

    return {
        'valid_loss': total_loss / divisor,
        'valid_mlm_loss': total_mlm_loss / divisor,
        'valid_aln': total_aln / divisor,
        'valid_unf': total_unf / divisor
    }


def simcl(device: Union[str, int], wd: Path, name: str, config: dict):
    distributed = config['n_gpus'] > 1
    device = torch.device(device)

    # If distributed training, initialise process group
    if distributed:
        dist.init_process_group(
            backend='nccl',
            rank=device.index,
            world_size=config['n_gpus']
        )

    # Load training data
    print('Loading data...')
    tokeniser = TOKENISERS[config['data']['tokeniser']]()
    train_dl = DATALOADERS[config['data']['dataloader']['name']](
        dataset=DATASETS[config['data']['dataset']](
            data=config['data']['train_path'],
            tokeniser=tokeniser
        ),
        distributed=distributed,
        num_replicas=config['n_gpus'],
        rank=device.index,
        **config['data']['dataloader']['config']
    )
    valid_dl = DATALOADERS[config['data']['dataloader']['name']](
        dataset=DATASETS[config['data']['dataset']](
            data=config['data']['valid_path'],
            tokeniser=tokeniser
        ),
        p_mask_random=0,
        p_mask_keep=0,
        **config['data']['dataloader']['config']
    )

    # Instantiate model
    print('Instantiating model...')
    model = MODELS[config['model']['name']](**config['model']['config'])
    model.to(device)
    model.load_state_dict(
        torch.load(config['model']['pretrain_state_dict_path'])
    )
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device])

    # Instantiate loss function and optimiser
    mlm_loss_fn = AdjustedCELoss(label_smoothing=0.1)
    simc_loss_fn = SimCLoss(**config['optim']['loss_config'])
    optimiser = Adam(
        params=model.parameters(),
        **config['optim']['optimiser_config']
    )

    # Evaluate model at pre-SimC learning state
    print('Evaluating pre-trained model state...')
    valid_metrics = validate(
        model=model,
        dl=valid_dl,
        mlm_loss_fn=mlm_loss_fn,
        simc_loss_fn=simc_loss_fn,
        device=device
    )
    metric_feedback(valid_metrics)

    metric_log = {
        0: {'loss': None, **valid_metrics}
    }

    # Go through epochs of training
    for epoch in range(1, config['n_epochs']+1):
        print(f'Starting epoch {epoch}...')

        if distributed:
            train_dl.sampler.set_epoch(epoch)
        
        print('Training...')
        train_metrics = train(
            model=model,
            dl=train_dl,
            simc_loss_fn=simc_loss_fn,
            optimiser=optimiser,
            device=device
        )
        metric_feedback(train_metrics)

        print('Validating...')
        valid_metrics = validate(
            model=model,
            dl=valid_dl,
            mlm_loss_fn=mlm_loss_fn,
            simc_loss_fn=simc_loss_fn,
            device=device
        )
        metric_feedback(valid_metrics)

        metric_log[epoch] = {**train_metrics, **valid_metrics}
    
    # Save results
    if distributed and device.index != 0:
        return
    
    print('Saving results...')
    save(
        wd=wd,
        save_name=name,
        model=model,
        log=metric_log,
        config=config
    )
    
    print('Done!')


def main(wd: Path, name: str, config: dict):
    # Distributed training
    if config['n_gpus'] > 1:
        print(
            f'Commencing distributed training on {config["n_gpus"]} CUDA '
            'devices...'
        )
        assert torch.cuda.device_count() > config["n_gpus"]
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '77777'
        mp.spawn(simcl, args=(wd, name, config), nprocs=config['n_gpus'])
        return

    # Single GPU traiing
    if config['n_gpus'] == 1:
        print('Commencing training on 1 CUDA device...')
        simcl(device=0, wd=wd, name=name, config=config)
        return

    # CPU training
    if config['n_gpus'] == 0:
        print('Commencing training on CPU...')
        simcl(device='cpu', wd=wd, name=name, config=config)
        return


if __name__ == '__main__':
    args = parse_command_line_arguments()

    if args.working_directory is None:
        wd = Path.cwd()
    else:
        wd = Path(args.working_directory).resolve()

    if args.name is None:
        name = datetime.now().strftime(r'%Y%m%d-%H%M%S')

    assert wd.is_dir()

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    main(wd=wd, name=name, config=config)