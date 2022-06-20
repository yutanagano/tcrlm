'''
pretrained_analysis.py
purpose: Run a myriad of analyses on a pretrained CDR3BERT model to analyse its
         performance, and what it has been able to learn from the data.
author: Yuta Nagano
ver: 0.1.1
'''


import argparse
from cgitb import handler
from lib2to3.pgen2.token import LPAR
from typing import Iterable
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm
from typing import Iterable, Union

from source.data_handling import Cdr3PretrainDataset, Cdr3PretrainDataLoader


# Some settings
MARKER_SIZE = 25
MARKER_ALPHA = 0.5


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


def generate_embeddings(pretrain_id: str, vdjdb_df: pd.DataFrame):
    cache_file_name = os.path.join(
        'cache',
        f'vdjdb_embeddings_{pretrain_id}.npy'
    )

    # If embeddings exist in cache then load that
    if os.path.isfile(cache_file_name):
        print(f'Loading embeddings for {pretrain_id} from cache...')
        return np.load(cache_file_name)
    
    # Otherwise create the embedding now
    print(f'Generating embeddings for {pretrain_id}...')

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'Using device {device}...')

    # Package vdjdb data into dataloader
    print('Formatting data...')
    df_formatted = vdjdb_df[['CDR3']].copy()
    df_formatted.loc[:,('frequency')] = 1
    ds = Cdr3PretrainDataset(
        data=df_formatted,
        p_mask=0
    )
    dl = Cdr3PretrainDataLoader(
        dataset=ds,
        batch_size=512,
        num_workers=4
    )

    # Load model
    print('Loading model...')
    model = torch.load(
        os.path.join('pretrain_runs', pretrain_id, 'pretrained.ptnn')
    ).bert.eval().to(device)

    # Run data through model and get resulting embeddings
    print('Obtaining model embeddings...')
    embs = []
    with torch.no_grad():
        for x, _ in tqdm(dl):
            x = x.to(device)
            embs.append(model.embed(x))
    embs = torch.cat(embs, dim=0).cpu().numpy()

    # Cache results
    print(f'Caching result as {cache_file_name}...')
    np.save(cache_file_name, embs)

    return embs


def generate_labels(groupids: pd.Series):
    # Get unique list
    uniques = groupids.sort_values().unique()

    # Assign colours to unique list, generate legend elements
    color_dict = dict()
    l_elements = []
    for i, id in enumerate(uniques):
        id_color = f'C{i}'
        if id is np.nan:
            id_color = '#000'
        color_dict[id] = id_color
        l_elements.append(
            Line2D(
                [0],[0],
                marker='o',markersize=5,
                color='#0000',
                markerfacecolor=id_color,markeredgecolor='#0000',
                label=id
            )
        )

    # Return colour list and legend elemends
    return groupids.map(lambda x: color_dict[x]), l_elements


def generate_2dvis(
    pca: np.ndarray,
    title: str,
    colours: Union[pd.Series, str] = 'C0',
    l_elements: Union[Iterable, None] = None,
    marker_size: int = MARKER_SIZE,
    marker_alpha: float = MARKER_ALPHA
):
    # Shuffle order of elements to draw in random order
    new_indices = np.arange(len(pca))
    np.random.shuffle(new_indices)
    pca = pca[new_indices]
    if type(colours) != str:
        colours = colours.iloc[new_indices]

    # Draw plot
    plt.figure(figsize=(10,10))
    plt.title(title)
    plt.scatter(
        pca[:,0], pca[:,1], c=colours, s=marker_size, alpha=marker_alpha
    )
    if l_elements is not None:
        plt.legend(handles=l_elements)
    plt.show()


def main(pretrain_id: str):
    # Load VDJDB sequences, either alpha or beta depending on the model
    print('Loading VDJDB data...')
    if 'alpha' in pretrain_id:
        df = pd.read_csv('data/vdjdb/vdjdb_homosapiens_alpha.csv')
    else:
        df = pd.read_csv('data/vdjdb/vdjdb_homosapiens_beta.csv')

    # Get model embeddings
    embs = generate_embeddings(pretrain_id, df)

    # Run embeddings through PCA to compress to 3 dimensions
    print("Compressing embeddings into 2d via PCA...")
    pca = PCA(n_components=3)
    embs3d = pca.fit_transform(embs)
    
    # Display embeddings
    generate_2dvis(
        pca=embs3d,
        title='Overall plot'
    )

    colours, legend = generate_labels(
        df['V'].str.extract(r'(TRBV\d+)', expand=False)\
            .str.replace(r'V(\d)$',r'V0\1',regex=True)
    )
    generate_2dvis(
        pca=embs3d,
        title='Coloured by V region',
        colours=colours,
        l_elements=legend
    )
    
    colours, legend = generate_labels(
        df['J'].str.extract(r'(TRBJ\d+)', expand=False)\
            .str.replace(r'J(\d)$',r'J0\1',regex=True)
    )
    generate_2dvis(
        pca=embs3d,
        title='Coloured by J region',
        colours=colours,
        l_elements=legend
    )

    colours, legend = generate_labels(
        df['MHC A'].str.extract(r'(HLA-[A-Z]+)', expand=False)
    )
    generate_2dvis(
        pca=embs3d,
        title='Coloured by MHC A restriction',
        colours=colours,
        l_elements=legend
    )

    generate_2dvis(
        pca=embs3d,
        title='Coloured by length',
        colours=df['CDR3'].str.len()
    )


if __name__ == '__main__':
    pretrain_id = parse_command_line_arguments()
    main(pretrain_id=pretrain_id)