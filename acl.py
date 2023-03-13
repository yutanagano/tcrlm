'''
An executable script to conduct simple contrastive learning on TCR models using
unlabelled TCR data.
'''


import argparse
from datetime import datetime
import json
from pathlib import Path
from src import modules
from src.modules.embedder import _MLMEmbedder
from src.datahandling import tokenisers
from src.datahandling.dataloaders import AutoContrastiveDataLoader
from src.datahandling.datasets import AutoContrastiveDataset
from src.metrics import (
    AdjustedCELoss,
    SimCLoss,
    SimCLoss2,
    AULoss,
    alignment_paired,
    uniformity,
    mlm_acc
)
from src.optim import AdamWithScheduling
from src.utils import save
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Union


MODELS = {
    'CDR3BERT_a': modules.CDR3BERT_a,
    'CDR3BERT_ap': modules.CDR3BERT_ap,
    'CDR3ClsBERT_ap': modules.CDR3ClsBERT_ap,
    'CDR3ClsBERT_ab': modules.CDR3ClsBERT_ab,
    'CDR3ClsBERT_apc': modules.CDR3ClsBERT_apc,
    'BVCDR3ClsBERT': modules.BVCDR3ClsBERT,
    'BCDRClsBERT': modules.BCDRClsBERT
}

TOKENISERS = {
    'CDR3Tokeniser': tokenisers.CDR3Tokeniser,
    'BCDR3Tokeniser': tokenisers.BCDR3Tokeniser,
    'BVCDR3Tokeniser': tokenisers.BVCDR3Tokeniser,
    'BCDRTokeniser': tokenisers.BCDRTokeniser
}

CONTRASTIVE_LOSSES = {
    'AULoss': AULoss,
    'SimCLoss': SimCLoss,
    'SimCLoss2': SimCLoss2
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
    model: _MLMEmbedder,
    dl: DataLoader,
    cont_loss_fn,
    mlm_loss_fn,
    optimiser,
    device
) -> dict:
    model.train()

    total_loss = 0
    total_lr = 0
    divisor = 0

    for x, x_prime, masked, target in tqdm(dl):
        num_samples = len(x)

        x = x.to(device)
        x_prime = x_prime.to(device)
        masked = masked.to(device)
        target = target.to(device)

        z = model.embed(x)
        z_prime = model.embed(x_prime)
        mlm_logits = model.mlm(masked)

        optimiser.zero_grad()
        loss = cont_loss_fn(z, z_prime) +\
            mlm_loss_fn(mlm_logits.flatten(0,1), target.view(-1))
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * num_samples
        total_lr += optimiser.lr * num_samples
        divisor += num_samples

    return {
        'loss': total_loss / divisor,
        'lr': total_lr / divisor
    }


@torch.no_grad()
def validate(
    model: _MLMEmbedder,
    dl: DataLoader,
    cont_loss_fn,
    mlm_loss_fn,
    device
) -> dict:
    total_cont_loss = 0
    total_mlm_loss = 0
    total_aln = 0
    total_unf = 0
    total_mlm_acc = 0
    divisor = 0

    for x, x_prime, masked, target in tqdm(dl):
        num_samples = len(x)

        x = x.to(device)
        x_prime = x_prime.to(device)
        masked = masked.to(device)
        target = target.to(device)

        model.train() # turn dropout on for contrastive eval, as it adds noise
        z = model.embed(x)
        z_prime = model.embed(x_prime)

        model.eval()
        mlm_logits = model.mlm(masked)

        cont_loss = cont_loss_fn(z, z_prime)
        mlm_loss = mlm_loss_fn(mlm_logits.flatten(0,1), target.view(-1))

        total_cont_loss += cont_loss.item() * num_samples
        total_mlm_loss += mlm_loss.item() * num_samples
        total_aln += alignment_paired(z, z_prime).item() * num_samples
        total_unf += uniformity(z).item() * num_samples
        total_mlm_acc += mlm_acc(mlm_logits, target) * num_samples
        divisor += num_samples

    return {
        'valid_cont_loss': total_cont_loss / divisor,
        'valid_mlm_loss': total_mlm_loss / divisor,
        'valid_aln': total_aln / divisor,
        'valid_unf': total_unf / divisor,
        'valid_mlm_acc': total_mlm_acc / divisor
    }


def simcl(device: Union[str, int], wd: Path, name: str, config: dict):
    device = torch.device(device)

    # Load training data
    print('Loading data...')
    tokeniser = TOKENISERS[config['data']['tokeniser']['class']](
        **config['data']['tokeniser']['config']
    )
    train_dl = AutoContrastiveDataLoader(
        dataset=AutoContrastiveDataset(
            data=config['data']['train_path'],
            tokeniser=tokeniser
        ),
        **config['data']['dataloader_config']
    )
    valid_dl = AutoContrastiveDataLoader(
        dataset=AutoContrastiveDataset(
            data=config['data']['valid_path'],
            tokeniser=tokeniser
        ),
        p_mask_random=0,
        p_mask_keep=0,
        **config['data']['dataloader_config']
    )

    # Instantiate model
    print('Instantiating model...')
    model = MODELS[config['model']['class']](**config['model']['config'])
    model.to(device)
    model.load_state_dict(
        torch.load(config['model']['pretrain_state_dict_path'])
    )

    # Instantiate loss function and optimiser
    mlm_loss_fn = AdjustedCELoss(label_smoothing=0.1)
    cont_loss_fn =\
        CONTRASTIVE_LOSSES[config['optim']['contrastive_loss']['name']](
            **config['optim']['contrastive_loss']['config']
        )
    optimiser = AdamWithScheduling(
        params=model.parameters(),
        d_model=config['model']['config']['d_model'],
        **config['optim']['optimiser_config']
    )

    # Evaluate model at pre-SimC learning state
    print('Evaluating pre-trained model state...')
    valid_metrics = validate(
        model=model,
        dl=valid_dl,
        cont_loss_fn=cont_loss_fn,
        mlm_loss_fn=mlm_loss_fn,
        device=device
    )
    metric_feedback(valid_metrics)

    metric_log = {
        0: {'loss': None, 'lr': None, **valid_metrics}
    }

    # Go through epochs of training
    for epoch in range(1, config['n_epochs']+1):
        print(f'Starting epoch {epoch}...')
        
        print('Training...')
        train_metrics = train(
            model=model,
            dl=train_dl,
            cont_loss_fn=cont_loss_fn,
            mlm_loss_fn=mlm_loss_fn,
            optimiser=optimiser,
            device=device
        )
        metric_feedback(train_metrics)

        print('Validating...')
        valid_metrics = validate(
            model=model,
            dl=valid_dl,
            cont_loss_fn=cont_loss_fn,
            mlm_loss_fn=mlm_loss_fn,
            device=device
        )
        metric_feedback(valid_metrics)

        metric_log[epoch] = {**train_metrics, **valid_metrics}
    
    # Save results    
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
    # Single GPU traiing
    if config['gpu']:
        print('Commencing training on 1 CUDA device...')
        simcl(device=0, wd=wd, name=name, config=config)
        return

    # CPU training
    print('Commencing training on CPU...')
    simcl(device='cpu', wd=wd, name=name, config=config)


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