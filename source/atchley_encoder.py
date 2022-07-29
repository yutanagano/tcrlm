'''
Simple implementations of a CDR3-encoding machine which uses atchley factors
(Atchley et al. 2005) to embed each individual amino acid in a CDR3 sequence,
then averages all values in each dimension to produce a fixed-sized vector
embedding for any CDR3 sequence. This is one of the baselines to which the CDR3
BERT model can be compared.
'''


import numpy as np
import pandas as pd


af = pd.read_csv('source/atchley_factors.csv',index_col='amino_acid')


def atchley_encode(cdr3: str) -> np.ndarray:
    embeddings = []
    for aa in cdr3: embeddings.append(af.loc[aa].to_numpy())

    embeddings = np.stack(embeddings)
    averaged = np.mean(embeddings,axis=0)

    return averaged