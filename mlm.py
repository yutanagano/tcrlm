'''
An executable script to conduct masked-language modelling on TCR models.
'''


import argparse
from datetime import datetime
import json
from pathlib import Path
from src import modules
from src.modules.embedder import _MLMEmbedder
from src.datahandling import tokenisers
from src.datahandling.dataloaders import MLMDataLoader
from src.datahandling.datasets import TCRDataset
from src.metrics import AdjustedCELoss, mlm_acc, mlm_topk_acc
from src.optim import AdamWithScheduling
from src.utils import save
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Union


MODELS = {
    'CDR3BERT_a': modules.CDR3BERT_a,
    'CDR3BERT_ap': modules.CDR3BERT_ap,
    'CDR3BERT_ac': modules.CDR3BERT_ac,
    'CDR3BERT_apc': modules.CDR3BERT_apc
}

TOKENISERS = {
    'ABCDR3Tokeniser': tokenisers.ABCDR3Tokeniser,
    'BCDR3Tokeniser': tokenisers.BCDR3Tokeniser
}


def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Masked-language modelling training loop.'
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
    loss_fn,
    optimiser,
    device
) -> dict:
    model.train()

    total_loss = 0
    total_lr = 0
    divisor = 0

    for x, y in tqdm(dl):
        num_samples = len(x)

        x = x.to(device)
        y = y.to(device)
        logits = model.mlm(x)

        optimiser.zero_grad()
        loss = loss_fn(logits.flatten(0,1), y.view(-1))
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
def validate(model: _MLMEmbedder, dl: DataLoader, loss_fn, device) -> dict:
    model.eval()

    total_loss = 0
    total_acc = 0
    total_top5_acc = 0
    divisor = 0

    for x, y in tqdm(dl):
        num_samples = len(x)

        x = x.to(device)
        y = y.to(device)

        logits = model.mlm(x)

        loss = loss_fn(logits.flatten(0,1), y.view(-1))

        total_loss += loss.item() * num_samples
        total_acc += mlm_acc(logits, y) * num_samples
        total_top5_acc += mlm_topk_acc(logits, y, 5) * num_samples
        divisor += num_samples

    return {
        'valid_loss': total_loss / divisor,
        'valid_acc': total_acc / divisor,
        'valid_top5_acc': total_top5_acc / divisor
    }


def mlm(device: Union[str, int], wd: Path, name: str, config: dict):
    device = torch.device(device)

    # Load training data
    print('Loading data...')
    tokeniser = TOKENISERS[config['data']['tokeniser']]()
    train_dl = MLMDataLoader(
        dataset=TCRDataset(
            data=config['data']['train_path'],
            tokeniser=tokeniser
        ),
        **config['data']['dataloader']['config']
    )
    valid_dl = MLMDataLoader(
        dataset=TCRDataset(
            data=config['data']['valid_path'],
            tokeniser=tokeniser
        ),
        p_mask_random=0,
        p_mask_keep=0,
        **config['data']['dataloader']['config']
    )

    # Instantiate model
    print('Instantiating model...')
    model = MODELS[config['model']['class']](**config['model']['config'])
    model.to(device)

    # Instantiate loss function and optimiser
    loss_fn = AdjustedCELoss(label_smoothing=0.1)
    optimiser = AdamWithScheduling(
        params=model.parameters(),
        d_model=config['model']['config']['d_model'],
        **config['optim']['optimiser_config']
    )

    metric_log = dict()

    # Go through epochs of training
    for epoch in range(1, config['n_epochs']+1):
        print(f'Starting epoch {epoch}...')
        
        print('Training...')
        train_metrics = train(model, train_dl, loss_fn, optimiser, device)
        metric_feedback(train_metrics)

        print('Validating...')
        valid_metrics = validate(model, valid_dl, loss_fn, device)
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
    # GPU traiing
    if config['gpu']:
        print('Commencing training on 1 CUDA device...')
        mlm(device=0, wd=wd, name=name, config=config)
        return

    # CPU training
    print('Commencing training on CPU...')
    mlm(device='cpu', wd=wd, name=name, config=config)


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