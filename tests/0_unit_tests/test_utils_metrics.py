import pytest
import source.utils.metrics as metrics
import torch


@pytest.mark.parametrize(
    ('x', 'expected'),
    (
        (
            torch.tensor(
                [
                    [2,6,4,21,21],
                    [1,2,3,4,5],
                    [8,3,21,21,21],
                    [21,21,21,21,21]
                ],
                dtype=torch.long
            ),
            torch.tensor([3,5,2,0])
        ),
        (
            torch.tensor(
                [
                    [0,1,2,3,4],
                    [0,1,2,3,21],
                    [0,1,2,21,21],
                    [0,1,21,21,21],
                    [0,21,21,21,21],
                    [21,21,21,21,21]
                ],
                dtype=torch.long
            ),
            torch.tensor([5,4,3,2,1,0])
        )
    )
)
def test_get_cdr3_lens(x, expected):
    result = metrics.get_cdr3_lens(x)
    assert torch.equal(result, expected)


class TestGetCdr3Third:
    @pytest.mark.parametrize(
        ('lens', 'third', 'expected'),
        (
            (
                torch.tensor([10,20,30]),
                0,
                (
                    torch.tensor([0,0,0]),
                    torch.tensor([3,7,10])
                )
            ),
            (
                torch.tensor([10,20,30]),
                1,
                (
                    torch.tensor([3,7,10]),
                    torch.tensor([7,13,20])
                )
            ),
            (
                torch.tensor([10,20,30]),
                2,
                (
                    torch.tensor([7,13,20]),
                    torch.tensor([10,20,30])
                )
            )
        )
    )
    def test_get_cdr3_third(self, lens, third, expected):
        result = metrics.get_cdr3_third(lens, third)
        assert torch.equal(result[0],expected[0])
        assert torch.equal(result[1],expected[1])


    @pytest.mark.parametrize(
        ('third'), (5, 10, -1)
    )
    def test_bad_third(self, third):
        with pytest.raises(RuntimeError):
            metrics.get_cdr3_third(torch.arange(3), third)


@pytest.mark.parametrize(
    ('x','start_indices','end_indices','expected'),
    (
        (
            torch.zeros(5,5),
            torch.tensor([0,0,0,0,0]),
            torch.tensor([1,2,1,3,3]),
            torch.tensor(
                [
                    [1,0,0,0,0],
                    [1,1,0,0,0],
                    [1,0,0,0,0],
                    [1,1,1,0,0],
                    [1,1,1,0,0]
                ]
            )
        ),
        (
            torch.zeros(5,5),
            torch.tensor([2,1,4,3,5]),
            torch.tensor([5,1,5,4,5]),
            torch.tensor(
                [
                    [0,0,1,1,1],
                    [0,0,0,0,0],
                    [0,0,0,0,1],
                    [0,0,0,1,0],
                    [0,0,0,0,0]
                ]
            )
        )
    )
)
def test_get_cdr3_partial_mask(x,start_indices,end_indices,expected):
    result = metrics.get_cdr3_partial_mask(x, start_indices, end_indices)
    assert torch.equal(result, expected)