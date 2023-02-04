import multiprocessing as mp
import pytest
from src.modules import *


mp.set_start_method('spawn')


@pytest.fixture
def cdr3bert_a_template():
    model = CDR3BERT_a(
        name='foobar',
        num_encoder_layers=2,
        d_model=4,
        nhead=2,
        dim_feedforward=16
    )
    return model


@pytest.fixture
def cdr3clsbert_apc_template():
    model = CDR3ClsBERT_apc(
        name='foobar',
        num_encoder_layers=2,
        d_model=4,
        nhead=2,
        dim_feedforward=16
    )
    return model