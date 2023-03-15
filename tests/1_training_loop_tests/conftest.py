import multiprocessing as mp
import pytest
from src.models import *


mp.set_start_method('spawn')


@pytest.fixture
def bcdr3bert_template():
    model = BCDR3BERT(
        name='foobar',
        num_encoder_layers=2,
        d_model=4,
        nhead=2,
        dim_feedforward=16
    )
    return model


@pytest.fixture
def cdr3clsbert_template():
    model = CDR3ClsBERT(
        name='foobar',
        num_encoder_layers=2,
        d_model=4,
        nhead=2,
        dim_feedforward=16
    )
    return model


@pytest.fixture
def bvcdr3bert_template():
    model = BVCDR3BERT(
        name='foobar',
        num_encoder_layers=2,
        d_model=4,
        nhead=2,
        dim_feedforward=16
    )
    return model