import numpy as np
import pytest
import source.benchmarking_algos as algos


# Positive tests
@pytest.mark.parametrize(
    'Algo',
    (
        algos.NegativeLevenshtein,
        algos.AtchleyCs
    )
)
def test_algo(Algo):
    assert(type(Algo.name) == str)
    assert(
        type(Algo.similarity_func('CAS','CSS')) in \
        (float, int, np.float64)
    )