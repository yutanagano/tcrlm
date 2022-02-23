'''
generate_train_stats_graph.py
purpose: Executable script to generate a graph visualising the training
         statistics for a particular version of the CDR3 BERT model.
author: Yuta Nagano
ver: 3.0.0
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
            'for a particular version of the CDR3 BERT model.'
    )
    parser.add_argument(
        'run_id',
        help='Specify the run ID for which to make ' + \
            'a training statistics visualisation.'
    )
    args = parser.parse_args()

    if not os.path.isdir(os.path.join('training_runs',args.run_id)):
        raise RuntimeError(
            f'No local directory (training_runs/{args.run_id}) could be found '\
            'which matches the run ID specified.'
        )

    return args.run_id


def load_training_stats(run_id: str) -> list:
    '''
    Given a run ID, load all training stats csvs available from that run using
    pandas, and return them all in a list.
    '''
    path_to_directory = os.path.join('training_runs',run_id)

    # Get a list of the contents of the train run directory
    contents = os.listdir(path_to_directory)

    # Only keep files that look like train_stats csvs
    train_stats = [os.path.join(path_to_directory, path) for path in contents \
        if re.fullmatch('^train_stats.*\.csv$', path)]
    
    # Raise error if no matches
    if len(train_stats) == 0:
        raise RuntimeError(
            f'No train stats csvs could be found for training run id {run_id}.'
        )
    
    # Load each identified train stat csv using pandas
    train_stat_dfs = [
        pd.read_csv(path,index_col='epoch') for path in train_stats
    ]

    return train_stat_dfs


def draw_figure(train_stat_dfs: list) -> matplotlib.figure.Figure:
    '''
    Given a set of pandas dataframes containing training statistics, draw a
    matplotlib figure that summarises the data, and return the figure object.
    '''
    # Take note of the number of epochs worth of data there is
    # The '-1' is because the last row of every dataframe is for jumbled loss (
    # not part of the training loop)
    epochs = len(train_stat_dfs[0]) - 1

    # Create figure
    fig = plt.figure(figsize=(8,8))

    # Create top panel (Loss)
    loss = plt.subplot(5,1,(1,2))
    loss.set_xticks(ticks=range(1,epochs + 1))
    loss.tick_params(
        axis='x',
        bottom=False,
        labelbottom=False
    )
    loss.set_title('Loss')

    # Create middle panel (Accuracy)
    acc = plt.subplot(5,1,(3,4))
    acc.set_xticks(ticks=range(1,epochs + 1))
    acc.tick_params(
        axis='x',
        bottom=False,
        labelbottom=False
    )
    acc.set_title('Accuracy')

    # Create bottom panel (Learning rate)
    lr = plt.subplot(5,1,5)
    lr.set_xticks(ticks=range(1,epochs + 1))
    lr.set_title('Average Learning Rate')

    # Plot top and middle panels (Loss and Accuracy)
    for df in train_stat_dfs:
        # Plot top panel (Loss)
        loss.plot(df['train_loss'],c='tab:blue')
        loss.plot(df['valid_loss'],c='tab:orange')

        # Plot middle panel (Accuracy)
        acc.plot(df['train_acc'],c='tab:blue')
        acc.plot(df['valid_acc'],c='tab:orange')

    # Plot bottom panel (Learning rate)
    lr.plot(train_stat_dfs[0]['avg_lr'],c='tab:green')

    # Create legends
    loss.legend(('training loss','validation loss'),loc='upper right')
    acc.legend(('training accuracy','validation accuracy'),loc='lower right')

    # Clean up figure
    fig.tight_layout()

    return fig


def main(run_id: str) -> None:
    '''
    Given a particular run ID, search for a local directory corresponding to
    that training run, fetch training stats data from that directory, produce
    a visualisation of the training statistics, and save the produced figure.
    '''
    # Read in the training stats
    stats = load_training_stats(run_id)

    # Create the figure
    fig = draw_figure(stats)

    # Save figure
    fig.savefig(os.path.join('training_runs',run_id,'train_stats.png'))


if __name__ == '__main__':
    # Set matplotlib stylesheet
    plt.style.use('seaborn')

    # Get run ID from the command line
    run_id = parse_command_line_arguments()
    
    # Run the main function
    main(run_id)