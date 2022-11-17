'''
An executable script to conduct masked-language modelling on TCR models.
'''


import argparse
from datetime import datetime
import json
from pathlib import Path
from src import modules
from src.modules.embedder import MLMEmbedder
from src.datahandling import tokenisers
from src.datahandling.dataloaders import TCRDataLoader
from src.datahandling.datasets import TCRDataset
from src.metrics import AdjustedCELoss, mlm_acc, mlm_topk_acc
from src.optim import AdamWithScheduling
from src.utils import save
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Union


MODELS = {
    'CDR3BERT_c': modules.CDR3BERT_c
}

TOKENISERS = {
    'CDR3Tokeniser': tokenisers.CDR3Tokeniser
}


def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Masked-language modelling training loop.'
    )
    parser.add_argument(
        '-d', '--working-directory',
        help='Path to tcr_embedder project working directory'
    )
    parser.add_argument(
        'config_path',
        help='Path to the training run config json file.'
    )
    return parser.parse_args()


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
        divisor += num_samples

    return {'loss': total_loss / divisor}


@torch.no_grad
def validate(model: MLMEmbedder, dl: DataLoader, loss_fn, device) -> dict:
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


def mlm(device: Union[str, int], wd: Path, config: dict):
    # Set device
    device = torch.device(device)

    # Load training data
    tokeniser = TOKENISERS[config['tokeniser']]()

    train_dl = TCRDataLoader(
        dataset=TCRDataset(
            data=config['train_data_path'],
            tokeniser=tokeniser
        ),
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
    model = MODELS[config['model']](**config['model_config']).to(device)

    # Instantiate loss function and optimiser
    loss_fn = AdjustedCELoss(label_smoothing=0.1)
    optimiser = AdamWithScheduling(
        params=model.parameters(),
        d_model=config['model_config']['d_model'],
        **config['optimiser_config']
    )

    metric_log = dict()

    # Go through epochs of training
    for epoch in range(1, config['n_epochs']+1):
        train_metrics = train(model, train_dl, loss_fn, optimiser, device)
        valid_metrics = validate(model, valid_dl, loss_fn, device)
        metric_log[epoch] = {**train_metrics, **valid_metrics}
    
    # Save results
    save(
        wd=wd,
        save_name=datetime.now().strftime(r"%Y%m%d_%H%M%S"),
        model=model,
        log=metric_log,
        config=config
    )


def main(wd: Path, config: dict):
    if config['n_gpus'] > 1:
        return

    if config['n_gpus'] == 1:
        mlm(device=0, wd=wd, config=config)
        return

    if config['n_gpus'] == 0:
        mlm(device='cpu', wd=wd, config=config)
        return


if __name__ == '__main__':
    args = parse_command_line_arguments()

    if args.working_directory is None:
        wd = Path.cwd()
    else:
        wd = Path(args.working_directory).resolve()

    assert wd.is_dir()

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    main(wd=wd, config=config)