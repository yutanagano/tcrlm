import pandas as pd
import pytest
import source.utils.datahandling as datahandling
import torch


@pytest.mark.parametrize(
    ('input_seq','expected'),
    (
        ('CAST', torch.tensor([1,0,15,16])),
        ('MESH', torch.tensor([10,3,15,6])),
        ('WILL', torch.tensor([18,7,9,9]))
    )
)
def test_tokenise(input_seq, expected):
    assert torch.equal(datahandling.tokenise(input_seq), expected)


class TestCheckDataframeFormat:
    def test_check_correct(self):
        df = pd.DataFrame(data=[], columns=['foo','bar'])
        datahandling.check_dataframe_format(
            dataframe=df,
            columns=['foo','bar']
        )
    

    def test_check_incorrect(self):
        df = pd.DataFrame(data=[], columns=['foo','bar'])
        with pytest.raises(RuntimeError):
            datahandling.check_dataframe_format(
                dataframe=df,
                columns=['bar','baz']
            )