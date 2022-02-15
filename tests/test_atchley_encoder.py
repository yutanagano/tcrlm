import numpy as np
import pandas as pd
import pytest
from source.atchley_encoder import atchley_encode


af = pd.read_csv('source/atchley_factors.csv',index_col='amino_acid')


# Positive tests
@pytest.mark.parametrize(
    'cdr3',
    [
        'CASSTGGLQGAFF',
        'CSVSYRAGSGNTEAFF',
        'CASSPHIAGAPYEQYF',
        'CASSVSTGNYGYTF',
        'CASSYDLGGAEDTQYF'
    ]
)
def test_atchley_encodings(cdr3):
    # Calculate what the embedding should be
    embeddings = [af.loc[aa].to_numpy() for aa in cdr3]
    embedding = np.stack(embeddings).mean(axis=0)

    # Ensure that the function produces the same
    assert(
        np.array_equal(embedding, atchley_encode(cdr3))
    )