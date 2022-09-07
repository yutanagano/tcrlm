import pytest
from source.nn import metrics
import torch
from torch.testing import assert_close


class TestAdjustedCELoss:
    def test_init(self):
        criterion = metrics.AdjustedCELoss(label_smoothing=0.5)

        assert criterion.label_smoothing == 0.5
        assert criterion.ignore_index == -2


    @pytest.mark.parametrize(
        ('y', 'expected'),
        (
            (torch.tensor([3,3]), torch.tensor(1.1864500045776367)),
            (torch.tensor([0,2]), torch.tensor(1.1330687999725342)),
            (torch.tensor([4,0]), torch.tensor(1.1398310661315918))
        )
    )
    def test_forward(self, y, expected):
        criterion = metrics.AdjustedCELoss()
        x = torch.tensor([[0.5,0.2,0.3],[0.3,0.3,0.4]])

        result = criterion(x, y)

        assert_close(result, expected)


    @pytest.mark.parametrize(
        'token', (1,5,-100)
    )
    def test_error_padding_tokens(self, token):
        criterion = metrics.AdjustedCELoss()
        x = torch.tensor([[0.5,0.2,0.3]])

        with pytest.raises(IndexError):
            criterion(x, torch.tensor([token]))


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
    assert_close(result, expected)


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
        assert_close(result[0],expected[0])
        assert_close(result[1],expected[1])


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
    assert_close(result, expected)


class TestPretrainMetrics:
    @pytest.mark.parametrize(
        ('logits', 'y', 'expected'),
        (
            (
                torch.tensor(
                    [
                        [
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                        ],
                        [
                            [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
                        ]
                    ],
                    dtype=torch.float
                ),
                torch.tensor(
                    [
                        [0,1,2,3,4],
                        [5,6,7,8,9]
                    ],
                    dtype=torch.long
                ),
                (torch.tensor(8) / torch.tensor(10)).item()
            ),
            (
                torch.tensor(
                    [
                        [
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                        ],
                        [
                            [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                        ],
                        [
                            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
                        ],
                        [
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
                        ]
                    ],
                    dtype=torch.float
                ),
                torch.tensor(
                    [
                        [0,1,2],
                        [3,4,5],
                        [6,7,8],
                        [9,21,21]
                    ],
                    dtype=torch.long
                ),
                (torch.tensor(6) / torch.tensor(10)).item()
            )
        )
    )
    def test_pretrain_accuracy(self,logits,y,expected):
        calculated = metrics.pretrain_accuracy(logits, y)
        assert_close(calculated, expected)


    @pytest.mark.parametrize(
        ('logits', 'y', 'k', 'expected'),
        (
            (
                torch.tensor(
                    [
                        [
                            [0.1,0.4,0,0,0,0.3,0,0,0,0.2,0,0,0,0,0,0,0,0,0,0],
                            [0,0.4,0.3,0.2,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0.1,0,0.2,0,0,0,0,0.3,0,0,0,0,0.4,0,0,0,0,0,0,0],
                            [0,0,0,0.3,0,0,0,0,0,0,0,0,0,0,0.2,0,0.4,0,0.1,0],
                            [0.2,0,0,0.3,0.4,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0,0]
                        ],
                        [
                            [0,0.3,0,0,0,0.2,0,0,0,0.1,0,0,0,0,0.4,0,0,0,0,0],
                            [0,0,0,0,0,0,0.4,0,0,0.3,0,0,0.2,0,0,0.1,0,0,0,0],
                            [0,0,0,0,0.4,0,0,0.3,0,0,0,0.2,0,0,0,0.1,0,0,0,0],
                            [0,0,0,0.3,0,0,0,0,0.4,0,0,0,0,0,0.2,0,0.1,0,0,0],
                            [0,0,0,0,0.4,0,0.3,0,0,0,0,0.2,0,0.1,0,0,0,0,0,0]
                        ]
                    ],
                    dtype=torch.float
                ),
                torch.tensor(
                    [
                        [0,1,2,3,4],
                        [5,6,7,8,9]
                    ],
                    dtype=torch.long
                ),
                3,
                (torch.tensor(8) / torch.tensor(10)).item()
            ),
            (
                torch.tensor(
                    [
                        [
                            [0.1,0.4,0,0,0,0.3,0,0,0,0.2,0,0,0,0,0,0,0,0,0,0],
                            [0,0.4,0.3,0.2,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0.1,0,0.2,0,0,0,0,0.3,0,0,0,0,0.4,0,0,0,0,0,0,0]
                        ],
                        [
                            [0,0,0,0.3,0,0,0,0,0,0,0,0,0,0,0.2,0,0.4,0,0.1,0],
                            [0.2,0,0,0.3,0.4,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0,0],
                            [0,0.3,0,0,0,0.2,0,0,0,0.1,0,0,0,0,0.4,0,0,0,0,0]
                        ],
                        [
                            [0,0,0.4,0,0,0,0,0,0,0.3,0,0,0.2,0,0,0.1,0,0,0,0],
                            [0,0,0,0,0.4,0,0,0.3,0,0,0,0.2,0,0,0,0.1,0,0,0,0],
                            [0,0,0,0.3,0,0,0,0,0.1,0,0,0,0,0,0.2,0,0,0,0.4,0]
                        ],
                        [
                            [0,0,0,0,0.4,0,0.3,0,0,0,0,0.2,0,0.1,0,0,0,0,0,0],
                            [0,0,0,0.3,0,0,0,0,0.4,0,0,0,0,0,0.2,0,0.1,0,0,0],
                            [0,0,0,0,0.4,0,0.3,0,0,0,0,0.2,0,0.1,0,0,0,0,0,0]
                        ]
                    ],
                    dtype=torch.float
                ),
                torch.tensor(
                    [
                        [0,1,2],
                        [3,4,5],
                        [6,7,8],
                        [9,21,21]
                    ],
                    dtype=torch.long
                ),
                3,
                (torch.tensor(6) / torch.tensor(10)).item()
            )
        )
    )
    def test_pretrain_topk_accuracy(self,logits,y,k,expected):
        calculated = metrics.pretrain_topk_accuracy(logits, y, k)
        assert_close(calculated, expected)


    @pytest.mark.parametrize(
        ('logits', 'x', 'y', 'third', 'expected'),
        (
            (
                torch.tensor(
                    [
                        [
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                        ],
                        [
                            [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
                        ]
                    ],
                    dtype=torch.float
                ),
                torch.tensor(
                    [
                        [0,1,2,3,4],
                        [5,6,7,21,21]
                    ]
                ),
                torch.tensor(
                    [
                        [0,1,2,3,4],
                        [5,6,21,21,21]
                    ],
                    dtype=torch.long
                ),
                0,
                (torch.tensor(2) / torch.tensor(3)).item()
            ),
            (
                torch.tensor(
                    [
                        [
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                        ],
                        [
                            [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
                        ]
                    ],
                    dtype=torch.float
                ),
                torch.tensor(
                    [
                        [0,1,2,3,4],
                        [5,6,7,21,21]
                    ]
                ),
                torch.tensor(
                    [
                        [0,1,2,3,4],
                        [5,6,21,21,21]
                    ],
                    dtype=torch.long
                ),
                1,
                (torch.tensor(2) / torch.tensor(2)).item()
            ),
            (
                torch.tensor(
                    [
                        [
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                        ],
                        [
                            [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
                        ]
                    ],
                    dtype=torch.float
                ),
                torch.tensor(
                    [
                        [0,1,2,3,4],
                        [5,6,7,21,21]
                    ]
                ),
                torch.tensor(
                    [
                        [0,1,2,3,4],
                        [5,6,21,21,21]
                    ],
                    dtype=torch.long
                ),
                2,
                (torch.tensor(1) / torch.tensor(2)).item()
            ),
            (
                torch.tensor(
                    [
                        [
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                        ],
                        [
                            [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
                        ]
                    ],
                    dtype=torch.float
                ),
                torch.tensor(
                    [
                        [0,1,2,3,4],
                        [5,6,7,21,21]
                    ]
                ),
                torch.tensor(
                    [
                        [0,1,2,21,21],
                        [5,6,21,21,21]
                    ],
                    dtype=torch.long
                ),
                2,
                None
            )
        )
    )
    def test_pretrain_accuracy_third(self,logits,x,y,third,expected):
        calculated = metrics.pretrain_accuracy_third(logits, x, y, third)

        if expected is None:
            assert calculated is None
            return
        
        assert_close(calculated, expected)


    @pytest.mark.parametrize(
        ('logits', 'x', 'y', 'k', 'third', 'expected'),
        (
            (
                torch.tensor(
                    [
                        [
                            [0.1,0.4,0,0,0,0.3,0,0,0,0.2,0,0,0,0,0,0,0,0,0,0],
                            [0,0.4,0.3,0.2,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0.1,0,0.2,0,0,0,0,0.3,0,0,0,0,0.4,0,0,0,0,0,0,0],
                            [0,0,0,0.3,0,0,0,0,0,0,0,0,0,0,0.2,0,0.4,0,0.1,0],
                            [0.2,0,0,0.4,0,0,0,0.3,0,0,0,0,0,0,0,0,0,0.1,0,0]
                        ],
                        [
                            [0,0.3,0,0,0,0.2,0,0,0,0.1,0,0,0,0,0.4,0,0,0,0,0],
                            [0,0,0,0,0,0,0.4,0,0,0.3,0,0,0.2,0,0,0.1,0,0,0,0],
                            [0,0,0,0,0.4,0,0,0.3,0,0,0,0.2,0,0,0,0.1,0,0,0,0],
                            [0,0,0,0.3,0,0,0,0,0.4,0,0,0,0,0,0.2,0,0.1,0,0,0],
                            [0,0,0,0,0.4,0,0.3,0,0,0,0,0.2,0,0.1,0,0,0,0,0,0]
                        ]
                    ],
                    dtype=torch.float
                ),
                torch.tensor(
                    [
                        [0,1,2,3,4],
                        [5,6,7,21,21]
                    ]
                ),
                torch.tensor(
                    [
                        [0,1,2,3,4],
                        [5,6,21,21,21]
                    ],
                    dtype=torch.long
                ),
                3,
                0,
                (torch.tensor(2) / torch.tensor(3)).item()
            ),
            (
                torch.tensor(
                    [
                        [
                            [0.1,0.4,0,0,0,0.3,0,0,0,0.2,0,0,0,0,0,0,0,0,0,0],
                            [0,0.4,0.3,0.2,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0.1,0,0.2,0,0,0,0,0.3,0,0,0,0,0.4,0,0,0,0,0,0,0],
                            [0,0,0,0.3,0,0,0,0,0,0,0,0,0,0,0.2,0,0.4,0,0.1,0],
                            [0.2,0,0,0.4,0,0,0,0.3,0,0,0,0,0,0,0,0,0,0.1,0,0]
                        ],
                        [
                            [0,0.3,0,0,0,0.2,0,0,0,0.1,0,0,0,0,0.4,0,0,0,0,0],
                            [0,0,0,0,0,0,0.4,0,0,0.3,0,0,0.2,0,0,0.1,0,0,0,0],
                            [0,0,0,0,0.4,0,0,0.3,0,0,0,0.2,0,0,0,0.1,0,0,0,0],
                            [0,0,0,0.3,0,0,0,0,0.4,0,0,0,0,0,0.2,0,0.1,0,0,0],
                            [0,0,0,0,0.4,0,0.3,0,0,0,0,0.2,0,0.1,0,0,0,0,0,0]
                        ]
                    ],
                    dtype=torch.float
                ),
                torch.tensor(
                    [
                        [0,1,2,3,4],
                        [5,6,7,21,21]
                    ]
                ),
                torch.tensor(
                    [
                        [0,1,2,3,4],
                        [5,6,21,21,21]
                    ],
                    dtype=torch.long
                ),
                3,
                1,
                (torch.tensor(2) / torch.tensor(2)).item()
            ),
            (
                torch.tensor(
                    [
                        [
                            [0.1,0.4,0,0,0,0.3,0,0,0,0.2,0,0,0,0,0,0,0,0,0,0],
                            [0,0.4,0.3,0.2,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0.1,0,0.2,0,0,0,0,0.3,0,0,0,0,0.4,0,0,0,0,0,0,0],
                            [0,0,0,0.3,0,0,0,0,0,0,0,0,0,0,0.2,0,0.4,0,0.1,0],
                            [0.2,0,0,0.4,0,0,0,0.3,0,0,0,0,0,0,0,0,0,0.1,0,0]
                        ],
                        [
                            [0,0.3,0,0,0,0.2,0,0,0,0.1,0,0,0,0,0.4,0,0,0,0,0],
                            [0,0,0,0,0,0,0.4,0,0,0.3,0,0,0.2,0,0,0.1,0,0,0,0],
                            [0,0,0,0,0.4,0,0,0.3,0,0,0,0.2,0,0,0,0.1,0,0,0,0],
                            [0,0,0,0.3,0,0,0,0,0.4,0,0,0,0,0,0.2,0,0.1,0,0,0],
                            [0,0,0,0,0.4,0,0.3,0,0,0,0,0.2,0,0.1,0,0,0,0,0,0]
                        ]
                    ],
                    dtype=torch.float
                ),
                torch.tensor(
                    [
                        [0,1,2,3,4],
                        [5,6,7,21,21]
                    ]
                ),
                torch.tensor(
                    [
                        [0,1,2,3,4],
                        [5,6,21,21,21]
                    ],
                    dtype=torch.long
                ),
                3,
                2,
                (torch.tensor(1) / torch.tensor(2)).item()
            ),
            (
                torch.tensor(
                    [
                        [
                            [0.1,0.4,0,0,0,0.3,0,0,0,0.2,0,0,0,0,0,0,0,0,0,0],
                            [0,0.4,0.3,0.2,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0.1,0,0.2,0,0,0,0,0.3,0,0,0,0,0.4,0,0,0,0,0,0,0],
                            [0,0,0,0.3,0,0,0,0,0,0,0,0,0,0,0.2,0,0.4,0,0.1,0],
                            [0.2,0,0,0.4,0,0,0,0.3,0,0,0,0,0,0,0,0,0,0.1,0,0]
                        ],
                        [
                            [0,0.3,0,0,0,0.2,0,0,0,0.1,0,0,0,0,0.4,0,0,0,0,0],
                            [0,0,0,0,0,0,0.4,0,0,0.3,0,0,0.2,0,0,0.1,0,0,0,0],
                            [0,0,0,0,0.4,0,0,0.3,0,0,0,0.2,0,0,0,0.1,0,0,0,0],
                            [0,0,0,0.3,0,0,0,0,0.4,0,0,0,0,0,0.2,0,0.1,0,0,0],
                            [0,0,0,0,0.4,0,0.3,0,0,0,0,0.2,0,0.1,0,0,0,0,0,0]
                        ]
                    ],
                    dtype=torch.float
                ),
                torch.tensor(
                    [
                        [0,1,2,3,4],
                        [5,6,7,21,21]
                    ]
                ),
                torch.tensor(
                    [
                        [0,1,2,21,21],
                        [5,6,21,21,21]
                    ],
                    dtype=torch.long
                ),
                3,
                2,
                None
            )
        )
    )
    def test_pretrain_topk_accuracy_third(self,logits,x,y,k,third,expected):
        calculated = metrics.pretrain_topk_accuracy_third(
            logits,
            x,
            y,
            k,
            third
        )

        if expected is None:
            assert calculated is None
            return

        assert_close(calculated, expected)


class TestFinetuneMetrics:
    @pytest.mark.parametrize(
        ('x', 'y', 'expected'),
        (
            (
                torch.tensor(
                    [
                        [1,0],
                        [1,0],
                        [1,0],
                        [0,1],
                        [0,1]
                    ],
                    dtype=torch.float
                ),
                torch.tensor(
                    [0,0,0,0,1],
                    dtype=torch.long
                ),
                (torch.tensor(4) / torch.tensor(5)).item()
            ),
            (
                torch.tensor(
                    [
                        [1,0],
                        [1,0],
                        [1,0],
                        [0,1],
                        [0,1]
                    ],
                    dtype=torch.float
                ),
                torch.tensor(
                    [1,1,0,0,0],
                    dtype=torch.long
                ),
                (torch.tensor(1) / torch.tensor(5)).item()
            )
        )
    )
    def test_finetune_accuracy(self,x,y,expected):
        calculated = metrics.finetune_accuracy(x, y)
        assert_close(calculated, expected)