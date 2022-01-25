'''
generate_train_stats_graph.py
purpose: Executable script to generate a graph visualising the training
         statistics for a particular version of the CDR3 BERT model.
author: Yuta Nagano
ver: 2.0.0
'''


import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd


if __name__ == '__main__':
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

    RUN_ID = args.run_id

    stats = pd.read_csv(os.path.join('training_runs',RUN_ID,'train_stats.csv'),
                        index_col='epoch')

    fig = plt.figure(figsize=(8,8))

    loss = plt.subplot(5,1,(1,2))
    loss.tick_params(
        axis='x',
        bottom=False,
        labelbottom=False
    )
    loss.set_title('Loss')

    loss.plot(stats['train_loss'],label='training loss')
    loss.plot(stats['valid_loss'],label='validation loss')
    loss.legend(loc='upper right')

    acc = plt.subplot(5,1,(3,4))
    acc.tick_params(
        axis='x',
        bottom=False,
        labelbottom=False
    )
    acc.set_title('Accuracy')

    acc.plot(stats['train_acc'],label='training accuracy')
    acc.plot(stats['valid_acc'],label='validation accuracy')
    acc.legend(loc='lower right')

    lr = plt.subplot(5,1,5)
    lr.set_xticks(
        ticks=range(1,21)
    )
    lr.set_title('Average Learning Rate')

    lr.plot(stats['avg_lr'],c='g')

    fig.tight_layout()

    plt.savefig(os.path.join('training_runs',RUN_ID,'train_stats.png'))