import io
import os
import pytest
import source.utils.misc as misc
import sys
import torch


def test_print_with_deviceid():
    out = io.StringIO()
    sys.stdout = out
    misc.print_with_deviceid('test', device=torch.device('cpu'))
    sys.stdout = sys.__stdout__

    assert out.getvalue() == '[cpu]: test\n'


def test_set_env_vars():
    try:
        del os.environ['MASTER_ADDR']
        del os.environ['MASTER_PORT']
    except(KeyError):
        pass

    misc.set_env_vars('localhost', '123456')

    assert os.getenv('MASTER_ADDR') == 'localhost'
    assert os.getenv('MASTER_PORT') == '123456'

    del os.environ['MASTER_ADDR']
    del os.environ['MASTER_PORT']


@pytest.mark.parametrize(
    ('l','expected'),
    (
        (
            [1,1,1,1,1,3,3,3,3,3,None,None],
            2
        ),
        (
            [0,1,2,3,4,5,6,7,8,9],
            4.5
        ),
        (
            [None, None, None],
            'n/a'
        ),
        (
            [],
            'n/a'
        )
    )
)
def test_dynamic_fmean(l, expected):
    result = misc.dynamic_fmean(l)
    assert result == expected