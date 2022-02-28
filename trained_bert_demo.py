'''
trained_bert_demo.py
purpose: Executable script to generate demo visualisations of a specified
         trained DR3 BERT model doing masked-residue modelling on a random
         selection of CDR3s from the testing set.
author: Yuta Nagano
ver: 3.0.1
'''


import argparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import pandas as pd
import random
import torch
from torch.nn import functional as F

from source.data_handling import tokenise, lookup

from hyperparams import hyperparams


def parse_command_line_arguments() -> str:
    '''
    Parse command line arguments from stdin (the ID of the training run in
    question) and return that as a string. Before returning, check that there is
    a local folder which corresponds to the specified run ID, and if not, raise
    an error.
    '''
    parser = argparse.ArgumentParser(
        description='Generate demo visualisations of a specified trained ' + \
            'CDR3 BERT model doing masked-residue modelling on a random ' + \
            'selection of CDR3s from the testing set.'
    )
    parser.add_argument(
        'run_id',
        help='Specify the run ID for which to make ' + \
            'the demo visualisations.'
    )
    args = parser.parse_args()

    if not os.path.isdir(os.path.join('training_runs',args.run_id)):
        raise RuntimeError(
            f'No local directory (training_runs/{args.run_id}) could be found '\
            'which matches the run ID specified.'
        )

    return args.run_id


@torch.no_grad()
def test_at_index(
    model: torch.nn.Module,
    cdr3: str,
    index: int,
    k: int = 5
) -> torch.tensor:
    # Ensure that the model is in evaluation mode
    model.eval()

    # Mask the residue at index
    masked = list(cdr3)
    masked[index] = '?'

    # Tokenise the cdr3
    tokenised = tokenise(masked)
    tokenised = tokenised.unsqueeze(0)

    mask = torch.zeros(tokenised.size())

    # Run the masked sequence through the model
    out = model(tokenised, mask)

    # Get the confidence distribution for the masked token prediction
    dist = F.softmax(out[0, index],dim=-1)

    return torch.topk(dist,k,dim=-1)


def generate_plot(
    model: torch.nn.Module,
    cdr3: str
) -> matplotlib.figure.Figure:
    '''
    Given a compatible pytorch model and a CDR3, generate a demo plot showcasing
    the model's understanding of the CDR3 amino acid sequence language by
    plotting the model's top 5 estimations of which amino acids should reside at
    each position, using the surrounding amino acid sequences as conditioning.
    '''
    # Generate a figure with appropriate dimensions
    l = len(cdr3)
    fig = plt.figure(figsize=(l,5))

    # Get a gridspec to make it easier to organise panels
    gs = GridSpec(5, l)

    # For each amino acid residue in the CDR3
    for i, aa in enumerate(cdr3):
        # Draw that letter at an appropriate position arond the top of the fig
        r = fig.add_subplot(gs[0,i])
        r.axis('off')
        r.text(0,0,aa,fontsize=48)

        # Get the model's top 5 predictions of the amino acid residue at that
        # position using the rest of the sequence as conditioning info
        confidences, indices = test_at_index(model, cdr3, i)
        confidences = list(reversed(confidences))
        guesses = list(reversed([lookup(idx.item()) for idx in indices]))

        # Plot a barchart of the top 5 guesses and their corresponding
        # confidence values
        h = fig.add_subplot(gs[1:,i])
        bars = h.barh(range(5),confidences)
        h.set_xlim(0,1)
        h.set_yticks(ticks=range(5),labels=guesses)
        h.tick_params(axis='y',which='both',length=0)

        # If any of the guesses are correct, highlight them in red
        for idx, label in enumerate(h.get_yticklabels()):
            if label.get_text() == aa:
                label.set_color('tab:red')
                bars[idx].set_color('tab:red')

    # Clean up the figure
    fig.tight_layout()

    return fig


def main(run_id: str):
    '''
    Given a particular run ID, load the trained model from that run, and use
    it to generate some demo plots of the model trying to guess the residues in
    a selection of CDR3s taken from the test set.
    '''

    # Load the trained model
    print('Loading model...')
    model = torch.load(
        os.path.join('training_runs',run_id,'trained_model.ptnn')
    )

    # Load the testing set
    print('Loading dataset...')
    ds = pd.read_csv(hyperparams['path_valid_data'])

    # Get a sample of CDR3s from the testing set
    cdr3s = random.sample(ds['CDR3'].to_list(),5)

    # Generate a demo plot for each CDR3
    for cdr3 in cdr3s:
        print(f'Generating plot for: {cdr3}...')

        fig = generate_plot(model, cdr3)

        # Save the generated figure
        fig.savefig(
            os.path.join('training_runs',run_id,f'{cdr3}.png')
        )
    
    print('Done!')


if __name__ == '__main__':
    # Set the random seed
    random.seed(42)

    # Parse command line arguments
    run_id = parse_command_line_arguments()

    # Execute main function
    main(run_id)