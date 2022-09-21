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
import pandas as pd
from pathlib import Path
from source.datahandling.datasets import Cdr3PretrainDataset
from source.datahandling.dataloaders import Cdr3PretrainDataLoader
from source.nn.models import Cdr3Bert
from source.utils import fileio
from source.utils import misc
from sklearn.decomposition import PCA
from sklearn import metrics
import torch
from tqdm import tqdm
from typing import Iterable, Union


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
    cache_file_name = \
        Path('.analysis_cache')/f'vdjdb_embeddings_{pretrain_id}.npy'

    # If embeddings exist in cache then load that
    if cache_file_name.is_file():
        print(f'Loading embeddings for {pretrain_id} from cache...')
        return np.load(cache_file_name)
    
    # Otherwise create the embedding now
    print(f'Generating embeddings for {pretrain_id}...')

    # Load necessary files
    print('Loading pretrain run hyperparameters...')
    hyperparams = fileio.parse_hyperparams(
        Path('pretrain_runs')/pretrain_id/'hyperparams.csv'
    )

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
        tokeniser=misc.instantiate_tokeniser(hyperparams),
        p_mask=0
    )
    dl = Cdr3PretrainDataLoader(
        dataset=ds,
        batch_size=512,
        num_workers=4
    )

    # Load model
    print('Loading model...')
    bert_state_dict = torch.load(
        Path('pretrain_runs')/pretrain_id/'bert_state_dict.pt'
    )

    model = Cdr3Bert(
        aa_vocab_size=hyperparams['aa_vocab_size'],
        num_encoder_layers=hyperparams['num_encoder_layers'],
        d_model=hyperparams['d_model'],
        nhead=hyperparams['nhead'],
        dim_feedforward=hyperparams['dim_feedforward'],
        activation=hyperparams['activation']
    ).to(device=device)
    model.load_state_dict(bert_state_dict)

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

    try:
        np.save(cache_file_name, embs)
    except(FileNotFoundError):
        Path('.analysis_cache').mkdir()
        np.save(cache_file_name, embs)

    return embs


def get_clustering_metric_table(pretrain_id: str):
    cache_file_name = \
        Path('pretrain_runs')/pretrain_id/'analysis'/'clustering_metrics.csv'

    # If embeddings exist in cache then load that
    if cache_file_name.is_file():
        print(f'Loading metric table for {pretrain_id} from cache...')
        return pd.read_csv(cache_file_name, index_col=0)
    
    # Otherwise return a new dataframe
    print(f"Generating clustering metric table for {pretrain_id}...")
    return pd.DataFrame(columns=['Silhouette','CH','DB'])


def get_control_table(pretrain_id: str, label_type: str):
    label_type = label_type.replace(' ', '_')

    cache_file_name = \
        Path('pretrain_runs')/pretrain_id/'analysis'/\
        f'clustering_metrics_control_{label_type}.csv'

    # If embeddings exist in cache then load that
    if cache_file_name.is_file():
        print(
            f'Loading {label_type} clustering control table for {pretrain_id} '
            'from cache...'
        )
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
    fig = plt.figure(figsize=(10,10))
    plt.title(title)
    plt.scatter(
        pca[:,0], pca[:,1], c=colours, s=marker_size, alpha=marker_alpha,
        linewidths=0
    )
    if l_elements is not None:
        plt.legend(handles=l_elements)

    return fig


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
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    plt.title(title)
    ax.scatter3D(
        pca[:,0], pca[:,1], pca[:,2],
        c=colours, s=marker_size, alpha=marker_alpha, linewidths=0
    )
    if l_elements is not None:
        plt.legend(handles=l_elements)
    
    return fig


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


def generate_control_table(
    pretrain_id: str,
    label_name: str,
    df: pd.DataFrame,
    embs: np.array
) -> pd.DataFrame:
    control_table = get_control_table(pretrain_id, label_name)
    for i in range(5):
        col_shuffled = df[label_name].sample(frac=1).reset_index(drop=True)
        compute_metrics(control_table, i, embs, col_shuffled)
    return control_table


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
    plt.show()

    # Create a folder to save these images in
    analysis_folder = Path('pretrain_runs')/pretrain_id/'analysis'
    try:
        analysis_folder.mkdir()
    except(FileExistsError):
        pass

    # According to V region
    colours, legend = generate_labels(df['V'])
    generate_2dvis(
        pca=embspca,
        title='Coloured by V region',
        colours=colours,
        l_elements=legend
    ).savefig(analysis_folder/'v.png')
    generate_3dvis(
        pca=embspca,
        title='Coloured by V region',
        colours=colours,
        l_elements=legend
    )
    plt.show()
    
    # According to J region
    colours, legend = generate_labels(df['J'])
    generate_2dvis(
        pca=embspca,
        title='Coloured by J region',
        colours=colours,
        l_elements=legend
    ).savefig(analysis_folder/'j.png')
    generate_3dvis(
        pca=embspca,
        title='Coloured by J region',
        colours=colours,
        l_elements=legend
    )
    plt.show()

    # According to MHC A restriction
    colours, legend = generate_labels(df['MHC A'])
    generate_2dvis(
        pca=embspca,
        title='Coloured by MHC A restriction',
        colours=colours,
        l_elements=legend
    ).savefig(analysis_folder/'mhca.png')
    generate_3dvis(
        pca=embspca,
        title='Coloured by MHC A restriction',
        colours=colours,
        l_elements=legend
    )
    plt.show()

    # According to MHC class
    colours, legend = generate_labels(df['MHC class'])
    generate_2dvis(
        pca=embspca,
        title='Coloured by MHC class',
        colours=colours,
        l_elements=legend
    ).savefig(analysis_folder/'mhc.png')
    generate_3dvis(
        pca=embspca,
        title='Coloured by MHC class',
        colours=colours,
        l_elements=legend
    )
    plt.show()

    # According to CDR3 length
    generate_2dvis(
        pca=embspca,
        title='Coloured by length',
        colours=df['CDR3'].str.len()
    ).savefig(analysis_folder/'cdr3len.png')
    generate_3dvis(
        pca=embspca,
        title='Coloured by length',
        colours=df['CDR3'].str.len()
    )
    plt.show()

    # According to epitope specificity
    high_confidence = df[df['Score'] == 3].copy()
    filtered_data = embspca[high_confidence.index]

    group_sizes = high_confidence.groupby('Epitope')['CDR3']\
                    .nunique().sort_values()
    significant_epitopes = group_sizes[-10:].index.to_list()
    high_confidence_viz = high_confidence[
        high_confidence['Epitope'].isin(significant_epitopes)
    ]
    filtered_data_viz = embspca[high_confidence_viz.index]

    colours, legend = generate_labels(
        high_confidence_viz['Epitope']
    )
    generate_2dvis(
        pca=filtered_data_viz,
        title='Coloured by epitope specificity',
        colours=colours,
        l_elements=legend,
        marker_size=100
    ).savefig(analysis_folder/'epitope.png')
    generate_3dvis(
        pca=filtered_data_viz,
        title='Coloured by epitope specificity',
        colours=colours,
        l_elements=legend,
        marker_size=100
    )
    plt.show()

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
    
    # Save the table
    clustering_metrics.to_csv(
        analysis_folder/f'clustering_metrics.csv'
    )

    print(clustering_metrics)

    # Compute 5 control values for each of the label types
    # Control for V regions
    print('Computing control metrics for V regions...')
    control_v = generate_control_table(
        pretrain_id=pretrain_id,
        label_name='V',
        df=df,
        embs=embspca
    )
    control_v.to_csv(
        analysis_folder/f'clustering_metrics_control_V.csv'
    )
    print(control_v)

    # Control for J regions
    print('Computing control metrics for J regions...')
    control_j = generate_control_table(
        pretrain_id=pretrain_id,
        label_name='J',
        df=df,
        embs=embspca
    )
    control_j.to_csv(
        analysis_folder/f'clustering_metrics_control_J.csv'
    )
    print(control_j)

    # Control for MHC A
    print('Computing control metrics for MHC A...')
    control_mhca = generate_control_table(
        pretrain_id=pretrain_id,
        label_name='MHC A',
        df=df,
        embs=embspca
    )
    control_mhca.to_csv(
        analysis_folder/f'clustering_metrics_control_MHC_A.csv'
    )
    print(control_mhca)

    # Control for MHC class
    print('Computing control metrics for MHC class...')
    control_mhc = generate_control_table(
        pretrain_id=pretrain_id,
        label_name='MHC class',
        df=df,
        embs=embspca
    )
    control_mhc.to_csv(
        analysis_folder/f'clustering_metrics_control_MHC_class.csv'
    )
    print(control_mhc)

    # Control for epitopes
    print('Computing control metrics for epitopes...')
    control_epitope = generate_control_table(
        pretrain_id=pretrain_id,
        label_name='Epitope',
        df=high_confidence,
        embs=filtered_data
    )
    control_epitope.to_csv(
        analysis_folder/f'clustering_metrics_control_Epitope.csv'
    )
    print(control_epitope)


if __name__ == '__main__':
    pretrain_id = parse_command_line_arguments()
    main(pretrain_id=pretrain_id)