import numpy as np
import pytest
import source.benchmarking_algos as algos


# Positive tests
@pytest.mark.parametrize(
    'Algo',
    (
        algos.NegativeLevenshtein(),
        algos.AtchleyCs(),
        algos.PretrainCdr3Bert(test_mode=True)
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
        algo = algos.PretrainCdr3Bert()

    
def test_pretrain_cdr3bert_bad_runid():
    with pytest.raises(FileNotFoundError):
        algo = algos.PretrainCdr3Bert('runid')