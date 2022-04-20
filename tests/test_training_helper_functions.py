import io
import os
import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import shutil
import sys

import source.training as training


# Positive tests
def test_create_new_training_run_directory():
    old_wd = os.getcwd()

    # Chdir into the tests directory
    os.chdir('tests')

    # Clean up if necessary
    if os.path.isdir('pretrain_runs'): shutil.rmtree('pretrain_runs')
    if os.path.isdir('finetune_runs'): shutil.rmtree('finetune_runs')

    # 1. Test that function creates pretrain run directories
    dirpath = training.create_training_run_directory('foo', mode='pretrain')
    assert(dirpath == 'pretrain_runs/foo')
    assert(os.path.isdir('pretrain_runs/foo'))
    dirpath = training.create_training_run_directory('bar', mode='pretrain')
    assert(dirpath == 'pretrain_runs/bar')
    assert(os.path.isdir('pretrain_runs/bar'))

    # 2. Test that function creates finetune run directories
    dirpath = training.create_training_run_directory('foo', mode='finetune')
    assert(dirpath == 'finetune_runs/foo')
    assert(os.path.isdir('finetune_runs/foo'))
    dirpath = training.create_training_run_directory('bar', mode='finetune')
    assert(dirpath == 'finetune_runs/bar')
    assert(os.path.isdir('finetune_runs/bar'))

    # 3. Test that function can dynamically alter run ID if clash
    dirpath = training.create_training_run_directory('foo', mode='pretrain')
    assert(dirpath == 'pretrain_runs/foo_1')
    assert(os.path.isdir('pretrain_runs/foo_1'))
    dirpath = training.create_training_run_directory('foo', mode='pretrain')
    assert(dirpath == 'pretrain_runs/foo_2')
    assert(os.path.isdir('pretrain_runs/foo_2'))

    # 4. Test that function can overwrite clash if in overwrite mode
    os.mkdir('pretrain_runs/foo/baz')
    dirpath = training.create_training_run_directory(
        'foo',
        mode='pretrain',
        overwrite=True
    )
    assert(dirpath == 'pretrain_runs/foo')
    assert(os.path.isdir('pretrain_runs/foo'))
    assert(not os.path.isdir('pretrain_runs/foo/baz'))

    # Return to normal working directory
    os.chdir(old_wd)


def test_write_hyperparameters():
    hp = {
        'foo': 'bar',
        'baz': 'foobar'
    }

    training.write_hyperparameters(hp, 'tests/data')

    with open('tests/data/hyperparams.txt', 'r') as f:
        for param in hp:
            line = f.readline()
            assert(line == f'{param}: {hp[param]}\n')
    
    os.remove('tests/data/hyperparams.txt')


def test_set_env_vars():
    training.set_env_vars('localhost', '123456')

    assert(os.getenv('MASTER_ADDR') == 'localhost')
    assert(os.getenv('MASTER_PORT') == '123456')


def test_print_with_deviceid():
    out = io.StringIO()
    sys.stdout = out
    training.print_with_deviceid('test', device=torch.device('cpu'))
    sys.stdout = sys.__stdout__

    assert(out.getvalue() == '[cpu]: test\n')


def test_compare_models():
    training.compare_models('tests/toy_models/identical', n_gpus=2)


def test_save_log():
    log = {
        0: {'foo': 'foo', 'bar': 'bar'},
        1: {'foo': 'foo', 'bar': 'bar'},
        2: {'foo': 'foo', 'bar': 'bar'}
    }

    expected = 'epoch,foo,bar\n0,foo,bar\n1,foo,bar\n2,foo,bar\n'

    training.save_log(log, 'tests/data', False, torch.device('cpu'))

    assert(os.path.isfile('tests/data/training_log.csv'))
    assert(open('tests/data/training_log.csv','r').read() == expected)

    os.remove('tests/data/training_log.csv')

    training.save_log(log, 'tests/data', True, torch.device('cpu'))

    assert(os.path.isfile('tests/data/training_log_cpu.csv'))
    assert(open('tests/data/training_log_cpu.csv','r').read() == expected)

    os.remove('tests/data/training_log_cpu.csv')


def test_save_model():
    model = torch.nn.Linear(3,3).to('cuda:0')

    def compare(model1: torch.nn.Module, model2: torch.nn.Module):
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if not torch.equal(p1, p2): return False
        return True

    training.save_model(
        model,
        'model',
        'tests/toy_models',
        False,
        torch.device('cpu'),
        False
    )

    assert(os.path.isfile('tests/toy_models/model.ptnn'))
    assert(compare(torch.load('tests/toy_models/model.ptnn'), model))

    os.remove('tests/toy_models/model.ptnn')


def test_save_model_distributed():
    if not torch.cuda.is_available():
        return

    model = torch.nn.Linear(3,3).to('cuda:0')

    training.set_env_vars('localhost', '12345')
    dist.init_process_group(
        backend='nccl',
        rank=0,
        world_size=1
    )
    distributed_model = DistributedDataParallel(model)

    def compare(model1: torch.nn.Module, model2: torch.nn.Module):
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if not torch.equal(p1, p2): return False
        return True

    training.save_model(
        distributed_model,
        'model',
        'tests/toy_models',
        True,
        torch.device('cuda:0'),
        False
    )

    assert(os.path.isfile('tests/toy_models/model.ptnn'))
    assert(compare(torch.load('tests/toy_models/model.ptnn'), model))

    os.remove('tests/toy_models/model.ptnn')

    # Save distributed and testing mode
    training.save_model(
        distributed_model,
        'model',
        'tests/toy_models',
        True,
        torch.device('cuda:0'),
        True
    )

    assert(os.path.isfile('tests/toy_models/model_cuda:0.ptnn'))
    assert(compare(torch.load('tests/toy_models/model_cuda:0.ptnn'), model))

    os.remove('tests/toy_models/model_cuda:0.ptnn')

    dist.destroy_process_group()


def test_parse_hyperparams():
    expected = {
        'path_train_data': 'tests/data/mock_unlabelled_data.csv',
        'path_valid_data': 'tests/data/mock_unlabelled_data.csv',
        'num_encoder_layers': 16,
        'd_model': 16,
        'nhead': 4,
        'dim_feedforward': 128,
        'activation': 'gelu',
        'train_batch_size': 6,
        'valid_batch_size': 6,
        'batch_optimisation': True,
        'lr': 0.001,
        'lr_decay': True,
        'optim_warmup': 5,
        'num_epochs': 3,
        'foo': False
    }
    
    hps = training.parse_hyperparams('tests/data/pretrain_hyperparams.csv')

    assert(hps == expected)


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
def test_pretrain_accuracy(logits,y,expected):
    calculated = training.pretrain_accuracy(logits, y)
    assert(calculated == expected)


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
def test_pretrain_topk_accuracy(logits,y,k,expected):
    calculated = training.pretrain_topk_accuracy(logits, y, k)
    assert(calculated == expected)


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
def test_get_cdr3_lens(x,expected):
    calculated = training._get_cdr3_lens(x)
    assert(torch.equal(calculated, expected))


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
def test_get_cdr3_third(lens,third,expected):
    calculated = training._get_cdr3_third(lens, third)
    assert(torch.equal(calculated[0],expected[0]))
    assert(torch.equal(calculated[1],expected[1]))


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
    calculated = training._get_cdr3_partial_mask(x, start_indices, end_indices)
    assert(torch.equal(calculated, expected))


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
def test_pretrain_accuracy_third(logits,x,y,third,expected):
    calculated = training.pretrain_accuracy_third(logits, x, y, third)
    assert(calculated == expected)


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
def test_pretrain_topk_accuracy_third(logits,x,y,k,third,expected):
    calculated = training.pretrain_topk_accuracy_third(logits, x, y, k, third)
    assert(calculated == expected)


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
    calculated = training.dynamic_fmean(l)
    assert(calculated == expected)


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
def test_finetune_accuracy(x,y,expected):
    calculated = training.finetune_accuracy(x, y)
    assert(calculated == expected)
    

# Negative tests
def test_create_new_training_run_directory_bad_mode():
    with pytest.raises(RuntimeError):
        training.create_training_run_directory('foo', mode='bar')


def test_compare_models_wrong_gpu_number():
    with pytest.raises(AssertionError):
        training.compare_models('tests/toy_models/identical', n_gpus=5)


def test_compare_models_nonidentical_models():
    with pytest.raises(RuntimeError):
        training.compare_models('tests/toy_models/different', n_gpus=2)


def test_parse_hyperparams_bad_path():
    with pytest.raises(RuntimeError):
        training.parse_hyperparams('csv_path')


def test_parse_hyperparams_bad_format():
    with pytest.raises(RuntimeError):
        training.parse_hyperparams('tests/data/bad_format.csv')


def test_parse_hyperparams_bad_types():
    with pytest.raises(RuntimeError):
        training.parse_hyperparams('tests/data/bad_types_hyperparams.csv')


def test_parse_hyperparams_bad_values():
    with pytest.raises(RuntimeError):
        training.parse_hyperparams('tests/data/bad_values_hyperparams.csv')


def test_get_cdr3_third_bad_third():
    with pytest.raises(RuntimeError):
        training._get_cdr3_third(torch.arange(3), 10)
    with pytest.raises(RuntimeError):
        training._get_cdr3_third(torch.arange(3), -1)