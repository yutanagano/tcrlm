'''
An executable script to conduct masked-language modelling on TCR models.
'''


import argparse
import json
from pathlib import Path
from src import modules
from src.datahandling import tokenisers
from src.datahandling.dataloaders import TCRDataLoader
from src.datahandling.datasets import TCRDataset
from src.metrics import AdjustedCELoss
from src.optim import AdamWithScheduling
import torch
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


def train(device: Union[str, int], wd: Path, config: dict):
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

    # Instantiate loss and optimiser
    loss = AdjustedCELoss(label_smoothing=0.1)
    optimiser = AdamWithScheduling(
        params=model.parameters(),
        d_model=config['model_config']['d_model'],
        **config['optimiser_config']
    )

    for epoch in range(1, config['n_epochs']+1):
        # TODO: do epoch through training data
        # TODO: compute validation metrics
        # TODO: log metrics
        pass
    
    # TODO: save results


def main(wd: Path, config: dict):
    if config['n_gpus'] > 1:
        return

    if config['n_gpus'] == 1:
        train(device=0, wd=wd, config=config)
        return

    if config['n_gpus'] == 0:
        train(device='cpu', wd=wd, config=config)
        return


if __name__ == '__main__':
    args = parse_command_line_arguments()

    if args.working_directory is None:
        wd = Path.cwd()
    else:
        wd = Path(args.working_directory).resolve()

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    main(wd=wd, config=config)