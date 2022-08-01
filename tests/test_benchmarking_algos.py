import numpy as np
import pytest
import source.benchmarking as benchmarking


# Positive tests
@pytest.mark.parametrize(
    'Algo',
    (
        benchmarking.NegativeLevenshtein(),
        benchmarking.AtchleyCs(),
        benchmarking.PretrainCdr3Bert(test_mode=True)
    )
)
def test_algo(Algo):
    assert(type(Algo.name) == str)
    assert(
        type(Algo.similarity_func('CAS','CSS')) in \
        (float, int, np.float64, np.float32)
    )


def test_pretrain_cdr3bert_no_runid():
    with pytest.raises(RuntimeError):
        algo = benchmarking.PretrainCdr3Bert()

    
def test_pretrain_cdr3bert_bad_runid():
    with pytest.raises(FileNotFoundError):
        algo = benchmarking.PretrainCdr3Bert('runid')