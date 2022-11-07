import io
import os
import pandas as pd
import pytest
from src.datahandling import tokenisers
from src.testing.mockups import MockDevice
from src.utils import misc
import sys


def test_print_with_deviceid():
    out = io.StringIO()
    sys.stdout = out
    misc.print_with_deviceid('test', device=MockDevice('cpu'))
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


class TestCheckDataframeFormat:
    def test_check_correct(self):
        df = pd.DataFrame(data=[], columns=['foo','bar'])
        misc.check_dataframe_format(
            dataframe=df,
            columns=['foo','bar']
        )
    

    def test_check_incorrect(self):
        df = pd.DataFrame(data=[], columns=['foo','bar'])
        with pytest.raises(RuntimeError):
            misc.check_dataframe_format(
                dataframe=df,
                columns=['bar','baz']
            )


class TestInstantiateTokeniser:
    def test_instantiate_aatokeniser(self):
        hyperparams = {
            'tokeniser_class': 'AaTokeniser',
            'tokeniser_hyperparams': '{"len_tuplet":1}'
        }

        result = misc.instantiate_tokeniser(hyperparameters=hyperparams)

        assert type(result) == tokenisers.AaTokeniser
        assert result.len_tuplet == 1


    def test_error_unrecognised_tokeniser_class(self):
        hyperparams = {
            'tokeniser_class': 'foobarbaz',
            'tokeniser_hyperparams': '{"foo":"bar"}'
        }

        with pytest.raises(RuntimeError):
            misc.instantiate_tokeniser(hyperparameters=hyperparams)