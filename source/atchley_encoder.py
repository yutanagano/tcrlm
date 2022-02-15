'''
atchley_encoder.py
purpose: This file contains code to implement a very simple CDR3-encoding
         machine which uses atchley factors (Atchley et al. 2005) to embed each
         individual amino acid in a CDR3 sequence, then averages all values in
         each dimension to produce a fixed-sized vector embedding for any CDR3
         sequence. This is one of the baselines to which the CDR3 BERT model
         can be compared.
author: Yuta Nagano
version: 1.0.0
'''


# Imports
import numpy as np
import pandas as pd


# Load atchley factor data
af = pd.read_csv('source/atchley_factors.csv',index_col='amino_acid')


# Main code
def atchley_encode(cdr3: str) -> np.ndarray:
    # Produces an averaged atchley-factor embedding for a single CDR3

    # Get atchley factor for each amino acid
    embeddings = []
    for aa in cdr3: embeddings.append(af.loc[aa].to_numpy())

    # Average across the zero-th dimension
    embeddings = np.stack(embeddings)
    averaged = np.mean(embeddings,axis=0)

    # Return the result
    return averaged