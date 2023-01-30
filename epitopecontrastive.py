'''
An executable script to conduct simple contrastive learning on TCR models using
a combination of unlabelled and epitope-labelled TCR data.
'''


import argparse
from datetime import datetime
import json
from pathlib import Path
from src import modules
from src.modules.embedder import MLMEmbedder
from src.datahandling import tokenisers
from src.datahandling.dataloaders import EpitopeAutoContrastiveSuperDataLoader
from src.datahandling.datasets import (
    AutoContrastiveDataset,
    EpitopeContrastiveDataset
)
from src.metrics import (
    AdjustedCELoss,
    SimCLoss,
    PosBackSimCLoss,
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
    'EpitopeContrastive_CDR3BERT_acp': modules.EpContCDR3BERT_acp,
    'EpitopeContrastive_BetaCDR3BERT_ap': modules.EpContBetaCDR3BERT_ap
}

TOKENISERS = {
    'CDR3Tokeniser': tokenisers.CDR3Tokeniser
}

AC_LOSSES = {
    'SimCLoss': SimCLoss
}

EC_LOSSES = {
    'PosBackSimCLoss': PosBackSimCLoss
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
    ec_loss_fn,
    ac_loss_fn,
    mlm_loss_fn,
    optimiser,
    device
) -> dict:
    model.train()

    total_loss = 0
    total_lr = 0
    divisor = 0

    for ac, ac_prime, ac_masked, ac_target, ec, ec_prime in tqdm(dl):
        num_samples = len(ac) + len(ec)

        ac = ac.to(device)
        ac_prime = ac_prime.to(device)
        ac_masked = ac_masked.to(device)
        ac_target = ac_target.to(device)
        ec = ec.to(device)
        ec_prime = ec_prime.to(device)

        z_ac = model.embed(ac)
        z_ac_prime = model.embed(ac_prime)
        mlm_logits = model.mlm(ac_masked)
        z_ec = model.embed(ec)
        z_ec_prime = model.embed(ec_prime)

        optimiser.zero_grad()
        loss = ec_loss_fn(z_ec, z_ec_prime, z_ac_prime) +\
            ac_loss_fn(z_ac, z_ac_prime) +\
            mlm_loss_fn(mlm_logits.flatten(0,1), ac_target.view(-1))
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
    model: MLMEmbedder,
    dl: DataLoader,
    ec_loss_fn,
    ac_loss_fn,
    mlm_loss_fn,
    device
) -> dict:
    total_ec_loss = 0
    total_ac_loss = 0
    total_mlm_loss = 0
    total_epitope_aln = 0
    total_auto_aln = 0
    total_unf = 0
    total_mlm_acc = 0
    ac_divisor = 0
    ec_divisor = 0

    for ac, ac_prime, ac_masked, ac_target, ec, ec_prime in tqdm(dl):
        num_ac_samples = len(ac)
        num_ec_samples = len(ec)

        ac = ac.to(device)
        ac_prime = ac_prime.to(device)
        ac_masked = ac_masked.to(device)
        ac_target = ac_target.to(device)
        ec = ec.to(device)
        ec_prime = ec_prime.to(device)

        model.train() # turn dropout on for autocontrastive eval
        z_ac = model.embed(ac)
        z_ac_prime = model.embed(ac_prime)

        model.eval()
        mlm_logits = model.mlm(ac_masked)
        z_ec = model.embed(ec)
        z_ec_prime = model.embed(ec_prime)

        ec_loss = ec_loss_fn(z_ec, z_ec_prime, z_ac_prime)
        ac_loss = ac_loss_fn(z_ac, z_ac_prime)
        mlm_loss = mlm_loss_fn(mlm_logits.flatten(0,1), ac_target.view(-1))

        total_ec_loss += ec_loss.item() * num_ec_samples
        total_ac_loss += ac_loss.item() * num_ac_samples
        total_mlm_loss += mlm_loss.item() * num_ac_samples
        total_epitope_aln +=\
            alignment_paired(z_ec, z_ec_prime).item() * num_ec_samples
        total_auto_aln +=\
            alignment_paired(z_ac, z_ac_prime).item() * num_ac_samples
        total_unf += uniformity(z_ac).item() * num_ac_samples
        total_mlm_acc += mlm_acc(mlm_logits, ac_target) * num_ac_samples
        ac_divisor += num_ac_samples
        ec_divisor += num_ec_samples

    return {
        'valid_ec_loss': total_ec_loss / ec_divisor,
        'valid_ac_loss': total_ac_loss / ac_divisor,
        'valid_mlm_loss': total_mlm_loss / ac_divisor,
        'valid_epitope_aln': total_epitope_aln / ec_divisor,
        'valid_auto_aln': total_auto_aln / ac_divisor,
        'valid_unf': total_unf / ac_divisor,
        'valid_mlm_acc': total_mlm_acc / ac_divisor
    }


def simcl(device: Union[str, int], wd: Path, name: str, config: dict):
    device = torch.device(device)

    # Load training data
    print('Loading data...')
    tokeniser = TOKENISERS[config['data']['tokeniser']]()
    train_dl = EpitopeAutoContrastiveSuperDataLoader(
        dataset_ac=AutoContrastiveDataset(
            data=config['data']['train_path']['autocontrastive'],
            tokeniser=tokeniser
        ),
        dataset_ec=EpitopeContrastiveDataset(
            data=config['data']['train_path']['epitope_contrastive'],
            tokeniser=tokeniser
        ),
        **config['data']['dataloader_config']
    )
    valid_dl = EpitopeAutoContrastiveSuperDataLoader(
        dataset_ac=AutoContrastiveDataset(
            data=config['data']['valid_path']['autocontrastive'],
            tokeniser=tokeniser
        ),
        dataset_ec=EpitopeContrastiveDataset(
            data=config['data']['valid_path']['epitope_contrastive'],
            tokeniser=tokeniser
        ),
        p_mask_random_ac=0,
        p_mask_keep_ac=0,
        **config['data']['dataloader_config']
    )

    # Instantiate model
    print('Instantiating model...')
    model = MODELS[config['model']['name']](
        contrastive_loss_type=config['optim']['autocontrastive_loss']['name'],
        **config['model']['config']
    )
    model.to(device)
    model.load_state_dict(
        torch.load(config['model']['pretrain_state_dict_path'])
    )

    # Instantiate loss function and optimiser
    mlm_loss_fn = AdjustedCELoss(label_smoothing=0.1)
    ac_loss_fn =\
        AC_LOSSES[config['optim']['autocontrastive_loss']['name']](
            **config['optim']['autocontrastive_loss']['config']
        )
    ec_loss_fn =\
        EC_LOSSES[config['optim']['epitope_contrastive_loss']['name']](
            **config['optim']['epitope_contrastive_loss']['config']
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
        ec_loss_fn=ec_loss_fn,
        ac_loss_fn=ac_loss_fn,
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
            ec_loss_fn=ec_loss_fn,
            ac_loss_fn=ac_loss_fn,
            mlm_loss_fn=mlm_loss_fn,
            optimiser=optimiser,
            device=device
        )
        metric_feedback(train_metrics)

        print('Validating...')
        valid_metrics = validate(
            model=model,
            dl=valid_dl,
            ec_loss_fn=ec_loss_fn,
            ac_loss_fn=ac_loss_fn,
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