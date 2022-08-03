import numpy as np
import pytest
from source.utils.atchleyencoder import atchley_encode


@pytest.mark.parametrize(
    ('input_seq', 'expected'),
    (
        (
            'CAST',
            np.mean(
                [
                    [-1.343,0.465,-0.862,-1.020,-0.255],
                    [-0.591,-1.302,-0.733,1.570,-0.146],
                    [-0.228,1.399,-4.760,0.670,-2.647],
                    [-0.032,0.326,2.213,0.908,1.313]
                ],
                axis=0
            )
        ),
        (
            'MESH',
            np.mean(
                [
                    [-0.663,-1.524,2.219,-1.005,1.212],
                    [1.357,-1.453,1.477,0.113,-0.837],
                    [-0.228,1.399,-4.760,0.670,-2.647],
                    [0.336,-0.417,-1.673,-1.474,-0.078]
                ],
                axis=0
            )
        ),
        (
            'WILL',
            np.mean(
                [
                    [-0.595,0.009,0.672,-2.128,-0.184],
                    [-1.239,-0.547,2.131,0.393,0.816],
                    [-1.019,-0.987,-1.505,1.266,-0.912],
                    [-1.019,-0.987,-1.505,1.266,-0.912]
                ],
                axis=0
            )
        )
    )
)
def test_atchley_encode(input_seq, expected):
    embedding = atchley_encode(aa_seq=input_seq)
    assert np.array_equal(embedding, expected)