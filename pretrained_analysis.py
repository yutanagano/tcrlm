'''
pretrained_analysis.py
purpose: Run a myriad of analyses on a pretrained CDR3BERT model to analyse its
         performance, and what it has been able to learn from the data.
author: Yuta Nagano
ver: 1.0.0
'''


import argparse
from typing import Iterable
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import metrics
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


def get_clustering_metric_table(pretrain_id: str):
    cache_file_name = os.path.join(
        'cache',
        f'clustering_metrics_{pretrain_id}.csv'
    )

    # If embeddings exist in cache then load that
    if os.path.isfile(cache_file_name):
        print(f'Loading embeddings for {pretrain_id} from cache...')
        return pd.read_csv(cache_file_name, index_col=0)
    
    # Otherwise return a new dataframe
    print(f"Generating clustering metric table for {pretrain_id}...")
    return pd.DataFrame(columns=['Silhouette','CH','DB'])


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


def generate_clustering_labels(groupids: pd.Series):
    # Get unique list
    uniques = groupids.sort_values().unique()

    # Assign integers to unique list
    id_dict = dict()
    for i, id in enumerate(uniques):
        id_dict[id] = i
    
    return groupids.map(lambda x: id_dict[x])


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
        pca[:,0], pca[:,1], c=colours, s=marker_size, alpha=marker_alpha,
        linewidths=0
    )
    if l_elements is not None:
        plt.legend(handles=l_elements)
    plt.show()


def generate_3dvis(
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
    ax = plt.axes(projection='3d')
    plt.title(title)
    ax.scatter3D(
        pca[:,0], pca[:,1], pca[:,2],
        c=colours, s=marker_size, alpha=marker_alpha, linewidths=0
    )
    if l_elements is not None:
        plt.legend(handles=l_elements)
    plt.show()


def compute_metrics(
    table: pd.DataFrame,
    row_name: str,
    embs: np.ndarray,
    df_col: pd.Series
):
    # Check if the named row exists already in the table
    if row_name in table.index:
        print(f"Metrics according to {df_col.name} exist, moving on...")
        return

    # Otherwise calculate what is necessary
    labels = generate_clustering_labels(df_col)

    # Calculate metrics
    print(f"Computing metrics according to {df_col.name}...")
    silhouette = metrics.silhouette_score(embs, labels)
    ch = metrics.calinski_harabasz_score(embs, labels)
    db = metrics.davies_bouldin_score(embs, labels)

    # Set row
    table.loc[row_name] = (silhouette, ch, db)


def main(pretrain_id: str):
    # Load VDJDB sequences, either alpha or beta depending on the model
    print('Loading VDJDB data...')
    if 'alpha' in pretrain_id:
        df = pd.read_csv('data/vdjdb/vdjdb_homosapiens_alpha.csv')
    else:
        df = pd.read_csv('data/vdjdb/vdjdb_homosapiens_beta.csv')
    
    # Reformat some columns
    df['V'] = df['V'].str.replace(r'.*(TR[AB]V\d+).*',r'\1',regex=True)
    df['J'] = df['J'].str.replace(r'.*(TR[AB]J\d+-?\d*).*',r'\1',regex=True)
    df['MHC A'] = df['MHC A'].str.replace(r'.*(HLA-[A-Z]+).*',r'\1',regex=True)

    # Get model embeddings
    embs = generate_embeddings(pretrain_id, df)

    # Run embeddings through PCA to compress to 3 dimensions
    print("Compressing embeddings into 3d via PCA...")
    pca = PCA()
    embspca = pca.fit_transform(embs)

    # Display ratios of variance explained by each of the PCAs
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    plt.figure()
    ax = plt.axes()
    ax.set_title('Data Variance Attributable to PCAs')
    ax.set_ylabel('Variance per PCA (bars)')
    axd = ax.twinx()
    axd.set_ylabel('Cumulative Variance (line)')
    axd.set_ylim(0,1.05)
    ax.bar(range(16), pca.explained_variance_ratio_)
    axd.plot(cumulative, c='C1')
    plt.show()
    
    # Display embeddings
    generate_3dvis(
        pca=embspca,
        title='Overall plot'
    )

    # According to V region
    colours, legend = generate_labels(df['V'])
    generate_3dvis(
        pca=embspca,
        title='Coloured by V region',
        colours=colours,
        l_elements=legend
    )
    
    # According to J region
    colours, legend = generate_labels(df['J'])
    generate_3dvis(
        pca=embspca,
        title='Coloured by J region',
        colours=colours,
        l_elements=legend
    )

    # According to MHC A restriction
    colours, legend = generate_labels(df['MHC A'])
    generate_3dvis(
        pca=embspca,
        title='Coloured by MHC A restriction',
        colours=colours,
        l_elements=legend
    )

    # According to MHC class
    colours, legend = generate_labels(df['MHC class'])
    generate_3dvis(
        pca=embspca,
        title='Coloured by MHC class',
        colours=colours,
        l_elements=legend
    )

    # According to CDR3 length
    generate_3dvis(
        pca=embspca,
        title='Coloured by length',
        colours=df['CDR3'].str.len()
    )

    # According to epitope specificity
    high_confidence = df[df['Score'] >= 2].copy()
    group_sizes = high_confidence.groupby('Epitope')['CDR3']\
                    .nunique().sort_values()
    significant_epitopes = group_sizes[-10:].index.to_list()
    high_confidence = high_confidence[
        high_confidence['Epitope'].isin(significant_epitopes)
    ]
    filtered_data = embspca[high_confidence.index]
    colours, legend = generate_labels(
        high_confidence['Epitope']
    )
    generate_3dvis(
        pca=filtered_data,
        title='Coloured by epitope specificity',
        colours=colours,
        l_elements=legend,
        marker_size=100
    )

    # Create table to keep track of clustering metrics
    print("Computing clustering metrics...")
    clustering_metrics = get_clustering_metric_table(pretrain_id)
    compute_metrics(clustering_metrics, 'V regions', embspca, df['V'])
    compute_metrics(clustering_metrics, 'J regions', embspca, df['J'])
    compute_metrics(clustering_metrics, 'MHC A', embspca, df['MHC A'])
    compute_metrics(clustering_metrics, 'MHC class', embspca, df['MHC class'])
    compute_metrics(
        clustering_metrics,
        'Epitope',
        filtered_data,
        high_confidence['Epitope']
    )
    
    # Cache the table
    clustering_metrics.to_csv(
        os.path.join('cache', f'clustering_metrics_{pretrain_id}.csv')
    )

    print(clustering_metrics)



if __name__ == '__main__':
    pretrain_id = parse_command_line_arguments()
    main(pretrain_id=pretrain_id)