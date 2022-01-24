import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import pandas as pd
import random
import argparse
import torch
from torch.nn import functional as F
from source.data_handling import CDR3Tokeniser


if __name__ == '__main__':
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


    RUN_ID = args.run_id


    random.seed(42)


    model = torch.load(os.path.join('training_runs',RUN_ID,'trained_model.ptnn'))
    ds = pd.read_csv('data/test.csv')
    tokeniser = CDR3Tokeniser()


    cdr3s = random.sample(ds['CDR3'].to_list(),5)


    @torch.no_grad()
    def test_at_index(model: torch.nn.Module, cdr3: str, index: int, k=5):
        # Ensure that the model is in evaluation mode
        model.eval()

        # Mask the residue at index
        masked = list(cdr3)
        masked[index] = '?'

        # Tokenise the cdr3
        tokenised = tokeniser.tokenise_in(masked)
        tokenised = tokenised.unsqueeze(0)

        mask = torch.zeros(tokenised.size())

        # Run the masked sequence through the model
        out = model(tokenised, mask)

        # Get the confidence distribution for the masked token prediction
        dist = F.softmax(out[0, index],dim=-1)

        return torch.topk(dist,k,dim=-1)

        
    def generate_plot(model: torch.nn.Module, cdr3: str):
        l = len(cdr3)
        fig = plt.figure(figsize=(l,5))
        gs = GridSpec(5, l)

        for i, aa in enumerate(cdr3):
            r = fig.add_subplot(gs[0,i])
            r.axis('off')
            r.text(0,0,aa,fontsize=48)

            confidences, indices = test_at_index(model, cdr3, i)
            confidences = list(reversed(confidences))
            guesses = list(reversed([tokeniser.lookup(idx.item()) for idx in indices]))

            h = fig.add_subplot(gs[1:,i])
            bars = h.barh(range(5),confidences)
            h.set_xlim(0,1)
            h.set_yticks(ticks=range(5),labels=guesses)
            h.tick_params(axis='y',which='both',length=0)
            for idx, label in enumerate(h.get_yticklabels()):
                if label.get_text() == aa:
                    label.set_color('tab:red')
                    bars[idx].set_color('tab:red')

        fig.tight_layout()
        
        plt.savefig(os.path.join('training_runs',RUN_ID,f'{cdr3}.png'))


    for cdr3 in cdr3s:
        generate_plot(model, cdr3)