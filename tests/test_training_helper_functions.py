import io
import os
import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import shutil
import sys

from source.training import create_training_run_directory, \
    write_hyperparameters, set_env_vars, print_with_deviceid, compare_models, \
    save_log, save_model


# Positive tests
def test_create_new_training_run_directory():
    old_wd = os.getcwd()

    # Chdir into the tests directory
    os.chdir('tests')

    # Clean up if necessary
    if os.path.isdir('pretrain_runs'): shutil.rmtree('pretrain_runs')
    if os.path.isdir('finetune_runs'): shutil.rmtree('finetune_runs')

    # 1. Test that function creates pretrain run directories
    dirpath = create_training_run_directory('foo', mode='pretrain')
    assert(dirpath == 'pretrain_runs/foo')
    assert(os.path.isdir('pretrain_runs/foo'))
    dirpath = create_training_run_directory('bar', mode='pretrain')
    assert(dirpath == 'pretrain_runs/bar')
    assert(os.path.isdir('pretrain_runs/bar'))

    # 2. Test that function creates finetune run directories
    dirpath = create_training_run_directory('foo', mode='finetune')
    assert(dirpath == 'finetune_runs/foo')
    assert(os.path.isdir('finetune_runs/foo'))
    dirpath = create_training_run_directory('bar', mode='finetune')
    assert(dirpath == 'finetune_runs/bar')
    assert(os.path.isdir('finetune_runs/bar'))

    # 3. Test that function can dynamically alter run ID if clash
    dirpath = create_training_run_directory('foo', mode='pretrain')
    assert(dirpath == 'pretrain_runs/foo_1')
    assert(os.path.isdir('pretrain_runs/foo_1'))
    dirpath = create_training_run_directory('foo', mode='pretrain')
    assert(dirpath == 'pretrain_runs/foo_2')
    assert(os.path.isdir('pretrain_runs/foo_2'))

    # 4. Test that function can overwrite clash if in overwrite mode
    os.mkdir('pretrain_runs/foo/baz')
    dirpath = create_training_run_directory(
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

    write_hyperparameters(hp, 'tests/data')

    with open('tests/data/hyperparams.txt', 'r') as f:
        for param in hp:
            line = f.readline()
            assert(line == f'{param}: {hp[param]}\n')
    
    os.remove('tests/data/hyperparams.txt')


def test_set_env_vars():
    set_env_vars('localhost', '123456')

    assert(os.getenv('MASTER_ADDR') == 'localhost')
    assert(os.getenv('MASTER_PORT') == '123456')


def test_print_with_deviceid():
    out = io.StringIO()
    sys.stdout = out
    print_with_deviceid('test', device=torch.device('cpu'))
    sys.stdout = sys.__stdout__

    assert(out.getvalue() == '[cpu]: test\n')


def test_compare_models():
    compare_models('tests/toy_models/identical', n_gpus=2)


def test_save_log():
    log = {
        0: {'foo': 'foo', 'bar': 'bar'},
        1: {'foo': 'foo', 'bar': 'bar'},
        2: {'foo': 'foo', 'bar': 'bar'}
    }

    expected = 'epoch,foo,bar\n0,foo,bar\n1,foo,bar\n2,foo,bar\n'

    save_log(log, 'tests/data', False, torch.device('cpu'))

    assert(os.path.isfile('tests/data/training_log.csv'))
    assert(open('tests/data/training_log.csv','r').read() == expected)

    os.remove('tests/data/training_log.csv')

    save_log(log, 'tests/data', True, torch.device('cpu'))

    assert(os.path.isfile('tests/data/training_log_cpu.csv'))
    assert(open('tests/data/training_log_cpu.csv','r').read() == expected)

    os.remove('tests/data/training_log_cpu.csv')


def test_save_model():
    model = torch.nn.Linear(3,3).to('cuda:0')

    def compare(model1: torch.nn.Module, model2: torch.nn.Module):
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if not torch.equal(p1, p2): return False
        return True

    save_model(
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

    set_env_vars('localhost', '12345')
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

    save_model(
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
    save_model(
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
    

# Negative tests
def test_create_new_training_run_directory_bad_mode():
    with pytest.raises(RuntimeError):
        create_training_run_directory('foo', mode='bar')


def test_compare_models_wrong_gpu_number():
    with pytest.raises(AssertionError):
        compare_models('tests/toy_models/identical', n_gpus=5)


def test_compare_models_nonidentical_models():
    with pytest.raises(RuntimeError):
        compare_models('tests/toy_models/different', n_gpus=2)