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
from src.datahandling.dataloaders import TCRDataLoader
from src.datahandling.datasets import TCRDataset
from src.metrics import SimCLoss, alignment_paired, uniformity
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
    loss_fn,
    optimiser,
    device
) -> dict:
    model.train()

    total_loss = 0
    divisor = 0

    for x in tqdm(dl):
        num_samples = len(x)

        x = x.to(device)
        z = model.embed(x)
        z_prime = model.embed(x)

        optimiser.zero_grad()
        loss = loss_fn(z, z_prime)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * num_samples
        divisor += num_samples

    return {'loss': total_loss / divisor}


@torch.no_grad()
def validate(model: MLMEmbedder, dl: DataLoader, loss_fn, device) -> dict:
    model.eval()

    total_loss = 0
    total_aln = 0
    total_unf = 0
    divisor = 0

    for x in tqdm(dl):
        num_samples = len(x)

        x = x.to(device)
        z = model.embed(x)
        z_prime = model.embed(x)

        loss = loss_fn(z, z_prime)

        total_loss += loss.item() * num_samples
        total_aln += alignment_paired(z, z_prime, alpha=2).item() * num_samples
        total_unf += uniformity(z, t=2).item() * num_samples
        divisor += num_samples

    return {
        'valid_loss': total_loss / divisor,
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
    tokeniser = TOKENISERS[config['tokeniser']]()
    train_dl = TCRDataLoader(
        dataset=TCRDataset(
            data=config['train_data_path'],
            tokeniser=tokeniser
        ),
        distributed=distributed,
        num_replicas=config['n_gpus'],
        rank=device.index,
        **config['dataloader_config']
    )
    valid_dl = TCRDataLoader(
        dataset=TCRDataset(
            data=config['valid_data_path'],
            tokeniser=tokeniser
        ),
        **config['dataloader_config']
    )

    # Instantiate model
    print('Instantiating model...')
    model = MODELS[config['model']](**config['model_config']).to(device)
    model.load_state_dict(torch.load(config['pretrain_state_dict_path']))
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device])

    # Instantiate loss function and optimiser
    loss_fn = SimCLoss(**config['loss_config'])
    optimiser = Adam(params=model.parameters(), **config['optimiser_config'])

    # Evaluate model at pre-SimC learning state
    valid_metrics = validate(model, valid_dl, loss_fn, device)

    metric_log = {0: {'loss': None, **valid_metrics}}

    # Go through epochs of training
    for epoch in range(1, config['n_epochs']+1):
        print(f'Starting epoch {epoch}...')

        if distributed:
            train_dl.sampler.set_epoch(epoch)
        
        print('Training...')
        train_metrics = train(model, train_dl, loss_fn, optimiser, device)
        metric_feedback(train_metrics)

        print('Validating...')
        valid_metrics = validate(model, valid_dl, loss_fn, device)
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