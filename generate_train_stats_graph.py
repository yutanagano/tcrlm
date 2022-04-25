'''
generate_train_stats_graph.py
purpose: Executable script to generate a graph visualising the training
         statistics for a particular version of the CDR3 BERT model.
author: Yuta Nagano
ver: 4.0.1
'''


import argparse
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import re


def parse_command_line_arguments() -> str:
    '''
    Parse command line arguments from stdin (the ID of the training run in
    question) and return that as a string. Before returning, check that there is
    a local folder which corresponds to the specified run ID, and if not, raise
    an error.
    '''
    parser = argparse.ArgumentParser(
        description='Generate a graph visualising the training statistics ' + \
            'for a particular training run of CDR3 BERT.'
    )
    parser.add_argument(
        '-p', '--pretrain-id',
        help='Specify the pretrain run ID for which to make ' + \
            'a training statistics visualisation.'
    )
    parser.add_argument(
        '-f', '--finetune-id',
        help='Specify the fine-tune run ID for which to make ' + \
            'a training statistics visualisation.'
    )
    
    args = parser.parse_args()

    return args.pretrain_id, args.finetune_id


def load_training_stats(pretrain_id: str = None, finetune_id: str = None) -> list:
    '''
    Given a pretrain or fine-tune training run ID, load all training stats csvs
    available from that run using pandas, and return them all in a list.
    '''
    if pretrain_id:
        if finetune_id:
            raise RuntimeError(
                'pretrain_id is mutually exclusive with finetune_id.'
            )
        path_to_directory = os.path.join('pretrain_runs',pretrain_id)
    elif finetune_id:
        path_to_directory = os.path.join('finetune_runs',finetune_id)
    else:
        raise RuntimeError(
            'Please specify either a pretrain or fine-tune training run ID.'
        )

    # Ensure that this directory exists
    if not os.path.isdir(path_to_directory):
        raise RuntimeError(
            f'No local directory ({path_to_directory}) could be found '\
            'which matches the run ID specified.'
        )

    # Get a list of the contents of the train run directory
    contents = os.listdir(path_to_directory)

    # Only keep files that look like train_stats csvs
    train_stats = [os.path.join(path_to_directory, path) for path in contents \
        if re.fullmatch('^training_log.*\.csv$', path)]
    
    # Raise error if no matches
    if len(train_stats) == 0:
        raise RuntimeError(
            f'No train stats csvs could be found in {path_to_directory}.'
        )
    
    # Load each identified train stat csv using pandas
    train_stat_dfs = [
        pd.read_csv(path,index_col='epoch') for path in train_stats
    ]

    return train_stat_dfs


def calculate_ticks(
    num_epochs: int,
    min_divisions: int = 10
) -> (list, list):
    '''
    Given the number of epochs, and the minimum number of divisions that must be
    made by the x-axis ticks, calculate the intervals for the major and minor
    ticks on the x-axis.
    '''
    tick = 1
    tick_min = 1
    tick_maj = 1
    counter = 0

    while num_epochs / tick > min_divisions:
        if counter % 2: tick *= 2
        else: tick *= 5

        counter+=1

        tick_min = tick_maj
        tick_maj = tick

    return (
        list(range(0,num_epochs+1,tick_maj)), 
        list(range(0,num_epochs+1,tick_min))
    )


def draw_figure(train_stat_dfs: list) -> matplotlib.figure.Figure:
    '''
    Given a set of pandas dataframes containing training statistics, draw a
    matplotlib figure that summarises the data, and return the figure object.
    '''
    # Take note of the number of epochs worth of data there is
    epochs = len(train_stat_dfs[0]['train_loss'].dropna())

    # Calculate the major and minor ticks of the x axis
    tick_maj, tick_min = calculate_ticks(epochs)

    # Create figure
    fig = plt.figure(figsize=(12,8))

    # Create top panel (Loss)
    loss = plt.subplot(2,2,1)
    loss.set_xticks(ticks=tick_maj)
    loss.set_xticks(ticks=tick_min,minor=True)
    loss.grid(which='minor',linewidth=0.5)
    loss.tick_params(
        axis='x',
        bottom=False,
        labelbottom=False
    )
    loss.set_title('Loss')

    # Create second panel (Accuracy)
    acc = plt.subplot(2,2,2)
    acc.set_ylim(bottom=0, top=1)
    acc.set_xticks(ticks=tick_maj)
    acc.set_xticks(ticks=tick_min,minor=True)
    acc.grid(which='minor',linewidth=0.5)
    acc.tick_params(
        axis='x',
        bottom=False,
        labelbottom=False
    )
    acc.set_title('Accuracy')

    # Create third panel (Accuracy)
    acc_thirds = plt.subplot(2,2,4)
    acc_thirds.set_ylim(bottom=0, top=1)
    acc_thirds.set_xticks(ticks=tick_maj)
    acc_thirds.set_xticks(ticks=tick_min,minor=True)
    acc_thirds.grid(which='minor',linewidth=0.5)
    acc_thirds.tick_params(
        axis='x',
        bottom=False,
        labelbottom=False
    )
    acc_thirds.set_title('Accuracy by CDR3 segment (only validation)')

    # Create bottom panel (Learning rate)
    lr = plt.subplot(2,2,3)
    lr.set_xticks(ticks=tick_maj)
    lr.set_xticks(ticks=tick_min,minor=True)
    lr.grid(which='minor',linewidth=0.5)
    lr.set_title('Average learning rate')

    # Plot top and middle panels (Loss and Accuracy)
    for df in train_stat_dfs:
        # Plot top panel (Loss)
        loss.plot(df['train_loss'],c='C0')
        loss.plot(df['valid_loss'],c='C1')

        # Plot second panel (Accuracy)
        acc.plot(df['train_acc'],c='C0')
        acc.plot(df['valid_acc'],c='C1')
        acc.plot(df['train_top5_acc'],c='C0',linestyle=':')
        acc.plot(df['valid_top5_acc'],c='C1',linestyle=':')

        # Plot third panel (Accuracy by CDR3 segment)
        acc_thirds.plot(df['valid_acc_third0'],c='C2')
        acc_thirds.plot(df['valid_acc_third1'],c='C3')
        acc_thirds.plot(df['valid_acc_third2'],c='C4')
        acc_thirds.plot(df['valid_top5_acc_third0'],c='C2',linestyle=':')
        acc_thirds.plot(df['valid_top5_acc_third1'],c='C3',linestyle=':')
        acc_thirds.plot(df['valid_top5_acc_third2'],c='C4',linestyle=':')

    # Plot bottom panel (Learning rate)
    lr.plot(train_stat_dfs[0]['avg_lr'],c='C2')

    # Create legends
    loss.legend(
        (
            'loss (training)',
            'loss (validation)'
        ),
        loc='upper right',
        prop={'size':8}
    )
    acc.legend(
        (
            'accuracy (training)',
            'top-5 accuracy (training)',
            'accuracy (validation)',
            'top-5 accuracy (validation)'
        ),
        loc='lower right',
        ncol=2,
        prop={'size':8}
    )
    acc_thirds.legend(
        (
            'accuracy (first third)',
            'accuracy (middle third)',
            'accuracy (final third)',
            'top-5 accuracy (first third)',
            'top-5 accuracy (middle third)',
            'top-5 accuracy (final third)'
        ),
        loc='lower right',
        ncol=2,
        prop={'size':8}
    )

    # Clean up figure
    fig.tight_layout()

    return fig


def main(pretrain_id: str, finetune_id: str) -> None:
    '''
    Given a particular run ID, search for a local directory corresponding to
    that training run, fetch training stats data from that directory, produce
    a visualisation of the training statistics, and save the produced figure.
    '''
    # Read in the training stats
    stats = load_training_stats(
        pretrain_id=pretrain_id,
        finetune_id=finetune_id
    )

    # Create the figure
    fig = draw_figure(stats)

    # Save figure
    if pretrain_id:
        parent_dir = 'pretrain_runs'
        run_id = pretrain_id
    else:
        parent_dir = 'finetune_runs'
        run_id = finetune_id

    fig.savefig(os.path.join(parent_dir,run_id,'train_stats.png'))


if __name__ == '__main__':
    # Set matplotlib stylesheet
    plt.style.use('seaborn')

    # Get run ID from the command line
    pretrain_id, finetune_id = parse_command_line_arguments()
    
    # Run the main function
    main(pretrain_id, finetune_id)