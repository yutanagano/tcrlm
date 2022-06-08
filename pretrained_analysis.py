'''
pretrained_analysis.py
purpose: Run a myriad of analyses on a pretrained CDR3BERT model to analyse its
         performance, and what it has been able to learn from the data.
author: Yuta Nagano
ver: 0.1.0
'''


import argparse
import os
import pandas as pd
from source.data_handling import Cdr3PretrainDataset, Cdr3PretrainDataLoader
import torch
from tqdm import tqdm

from source.training import pretrain_accuracy


def parse_command_line_arguments() -> str:
    parser = argparse.ArgumentParser(
        description='Run a myriad of analyses on a pretrained CDR3BERT '
            'model to analyse its performance, and what it has been able to '
            'learn from the data.'
    )
    parser.add_argument(
        'pretrain_id',
        help='Specify the pretrain run ID for the model to analyse.'
    )
    args = parser.parse_args()

    return args.pretrain_id


def main(
    pretrain_id: str
):
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Load model
    print('Loading model...')
    model = torch.load(
        os.path.join('pretrain_runs', pretrain_id, 'pretrained.ptnn')
    ).bert.eval()

    # Load VDJDB sequences, either alpha or beta depending on the model
    print('Loading VDJDB data...')
    if 'alpha' in pretrain_id:
        df = pd.read_csv('data/vdjdb/vdjdb_homosapiens_alpha.csv')
    else:
        df = pd.read_csv('data/vdjdb/vdjdb_homosapiens_beta.csv')

    # Wrap VDJDB data around a dataset and dataloader
    print('Reformatting data...')
    df = df[['CDR3']].drop_duplicates()
    df[['frequency']] = 1

    ds = Cdr3PretrainDataset(
        data=df,
        p_mask=0
    )
    dl = Cdr3PretrainDataLoader(
        dataset=ds,
        batch_size=512,
        num_workers=4,
        batch_optimisation=True
    )

    # Run data through model and get resulting embeddings
    print('Obtaining model embeddings...')
    embs = []

    with torch.no_grad():
        for x, _ in tqdm(dl):
            x = x.to(device)
            embs.append(model.embed(x))
    
    print('Cleaning up results...')
    embs = torch.cat(embs, dim=0)

    # Run embeddings through PCA to compress to 2 dimensions
    return


if __name__ == '__main__':
    pretrain_id = parse_command_line_arguments()
    main(pretrain_id=pretrain_id)