from finetune import main as main_f
import multiprocessing as mp
import pandas as pd
from pathlib import Path
import pickle
from pretrain import main as main_p
import pytest
from shutil import copy
from source.nn.models import Cdr3BertPretrainWrapper, Cdr3BertFineTuneWrapper
from source.utils.datahandling import check_dataframe_format
from source.utils.fileio import parse_hyperparams
import torch
from typing import Any
from warnings import warn


mp.set_start_method('spawn')


@pytest.fixture(scope='module')
def pretrain_hyperparams_path():
    return 'tests/resources/hyperparams/pretrain.csv'


@pytest.fixture(scope='module')
def finetune_hyperparams_path():
    return 'tests/resources/hyperparams/finetune.csv'


@pytest.fixture(scope='module')
def expected_pretrain_hyperparams(pretrain_hyperparams_path):
    hyperparams = parse_hyperparams(
        csv_path=pretrain_hyperparams_path
    )
    return hyperparams


@pytest.fixture(scope='module')
def expected_finetune_hyperparams(finetune_hyperparams_path):
    hyperparams = parse_hyperparams(
        csv_path=finetune_hyperparams_path
    )
    return hyperparams


@pytest.fixture(scope='module')
def expected_pretrain_log_cols():
    cols = [
        'epoch',
        'train_loss',
        'train_acc',
        'train_top5_acc',
        'train_acc_third0',
        'train_top5_acc_third0',
        'train_acc_third1',
        'train_top5_acc_third1',
        'train_acc_third2',
        'train_top5_acc_third2',
        'avg_lr',
        'epoch_time',
        'valid_loss',
        'valid_acc',
        'valid_top5_acc',
        'valid_acc_third0',
        'valid_top5_acc_third0',
        'valid_acc_third1',
        'valid_top5_acc_third1',
        'valid_acc_third2',
        'valid_top5_acc_third2',
        'jumble_loss',
        'jumble_acc',
        'jumble_top5_acc',
        'jumble_acc_third0',
        'jumble_top5_acc_third0',
        'jumble_acc_third1',
        'jumble_top5_acc_third1',
        'jumble_acc_third2',
        'jumble_top5_acc_third2'
    ]
    return cols


@pytest.fixture(scope='module')
def expected_finetune_log_cols():
    cols = [
        'epoch',
        'train_loss',
        'train_acc',
        'avg_lr',
        'epoch_time',
        'valid_loss',
        'valid_acc'
    ]
    return cols


@pytest.fixture(scope='module')
def expected_pretrain_parameters_template():
    with open('tests/resources/parameters/pretrain.pickle', 'rb') as f:
        params = pickle.load(f)
    return params


@pytest.fixture(scope='module')
def expected_finetune_parameters_template():
    with open('tests/resources/parameters/finetune.pickle', 'rb') as f:
        params = pickle.load(f)
    return params


@pytest.fixture(scope='function')
def tmp_finetune_working_dir(tmp_path):
    pretrain_runs_dir = tmp_path / 'pretrain_runs'
    pretrain_runs_dir.mkdir()

    test_dir = pretrain_runs_dir / 'test'
    test_dir.mkdir()

    copy('tests/resources/models/pretrained.ptnn', test_dir)

    return tmp_path


def check_hyperparam_record(training_run_dir: Path, expected_hyperparams: dict):
    expected_location = training_run_dir / 'hyperparams.txt'

    assert expected_location.is_file()
    
    with open(expected_location, 'r') as f:
        for param in expected_hyperparams:
            line = f.readline()
            assert line == f'{param}: {expected_hyperparams[param]}\n'


def check_training_log(
    training_run_dir: Path,
    expected_name: str,
    expected_columns: list,
    expected_length: int
):
    expected_location = training_run_dir / expected_name

    assert expected_location.is_file()
    
    log = pd.read_csv(expected_location)

    assert len(log) == expected_length
    check_dataframe_format(dataframe=log, columns=expected_columns)


def check_saved_model(
    training_run_dir: Path,
    expected_name: str,
    expected_type: Any,
    expected_params_template: tuple
):
    expected_location = training_run_dir / expected_name

    assert expected_location.is_file()

    model = torch.load(expected_location)

    assert type(model) == expected_type

    for p1, p2 in zip(model.parameters(), expected_params_template):
        assert p1.size() == p2.size()


def check_models_equivalent(training_run_dir: Path, n_gpus: int) -> None:
    paths_to_models = [
        path for path in training_run_dir.iterdir() if path.suffix == '.ptnn'
    ]

    assert len(paths_to_models) == n_gpus

    models = [torch.load(path) for path in paths_to_models]

    def models_equivalent(model1: torch.nn.Module, model2: torch.nn.Module):
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if not torch.equal(p1, p2):
                return False
        return True
        
    for model2 in models[1:]:
        assert models_equivalent(models[0], model2)


class TestPretrainLoop:
    @pytest.mark.parametrize(
        ('n_gpus'), (0, 1)
    )
    def test_pretrain_loop(
        self,
        tmp_path,
        pretrain_hyperparams_path,
        expected_pretrain_hyperparams,
        expected_pretrain_log_cols,
        expected_pretrain_parameters_template,
        n_gpus
    ):
        if n_gpus == 1 and not torch.cuda.is_available():
            warn('Pretrain GPU test skipped due to hardware limitations.')
            return

        # Run the training loop in a separate process
        p = mp.Process(
            target=main_p,
            kwargs={
                'working_directory': tmp_path,
                'run_id': 'test',
                'hyperparams_path': pretrain_hyperparams_path,
                'n_gpus': n_gpus,
                'test_mode': True
            }
        )
        p.start()
        p.join()

        expected_training_run_dir = tmp_path / 'pretrain_runs' / 'test'

        assert expected_training_run_dir.is_dir()

        check_hyperparam_record(
            training_run_dir=expected_training_run_dir,
            expected_hyperparams=expected_pretrain_hyperparams
        )
        check_training_log(
            training_run_dir=expected_training_run_dir,
            expected_name='training_log.csv',
            expected_columns=expected_pretrain_log_cols,
            expected_length=4 # 3 epochs + jumble row
        )
        check_saved_model(
            training_run_dir=expected_training_run_dir,
            expected_name='pretrained.ptnn',
            expected_type=Cdr3BertPretrainWrapper,
            expected_params_template=expected_pretrain_parameters_template
        )


    def test_pretrain_loop_distributed(
        self,
        tmp_path,
        pretrain_hyperparams_path,
        expected_pretrain_hyperparams,
        expected_pretrain_log_cols,
        expected_pretrain_parameters_template
    ):
        n_gpus = torch.cuda.device_count()
        if n_gpus <= 1:
            warn(
                'Pretrain distributed test skipped due to hardware '
                'limitations.'
            )
            return

        # Run the training loop in a separate process
        p = mp.Process(
            target=main_p,
            kwargs={
                'working_directory': tmp_path,
                'run_id': 'test',
                'hyperparams_path': pretrain_hyperparams_path,
                'n_gpus': n_gpus,
                'test_mode': True
            }
        )
        p.start()
        p.join()

        expected_training_run_dir = tmp_path / 'pretrain_runs' / 'test'

        assert expected_training_run_dir.is_dir()

        check_hyperparam_record(
            training_run_dir=expected_training_run_dir,
            expected_hyperparams=expected_pretrain_hyperparams
        )
        for i in range(n_gpus):
            check_training_log(
                training_run_dir=expected_training_run_dir,
                expected_name=f'training_log_cuda_{i}.csv',
                expected_columns=expected_pretrain_log_cols,
                expected_length=4 # 3 epochs + jumble row
            )
        check_saved_model(
            training_run_dir=expected_training_run_dir,
            expected_name='pretrained_cuda_0.ptnn',
            expected_type=Cdr3BertPretrainWrapper,
            expected_params_template=expected_pretrain_parameters_template
        )
        check_models_equivalent(
            training_run_dir=expected_training_run_dir,
            n_gpus=n_gpus
        )


class TestFinetuneLoop:
    @pytest.mark.parametrize(
        ('n_gpus'), (0, 1)
    )
    def test_finetune_loop(
        self,
        tmp_finetune_working_dir,
        finetune_hyperparams_path,
        expected_finetune_hyperparams,
        expected_finetune_log_cols,
        expected_finetune_parameters_template,
        n_gpus
    ):
        if n_gpus == 1 and not torch.cuda.is_available():
            warn('Finetune GPU test skipped due to hardware limitations.')
            return

        # Run the training loop in a separate process
        p = mp.Process(
            target=main_f,
            kwargs={
                'working_directory': tmp_finetune_working_dir,
                'run_id': 'test',
                'hyperparams_path': finetune_hyperparams_path,
                'n_gpus': n_gpus,
                'test_mode': True
            }
        )
        p.start()
        p.join()

        expected_training_run_dir = tmp_finetune_working_dir / \
            'finetune_runs' / 'test'

        assert expected_training_run_dir.is_dir()

        check_hyperparam_record(
            training_run_dir=expected_training_run_dir,
            expected_hyperparams=expected_finetune_hyperparams
        )
        check_training_log(
            training_run_dir=expected_training_run_dir,
            expected_name='training_log.csv',
            expected_columns=expected_finetune_log_cols,
            expected_length=3
        )
        check_saved_model(
            training_run_dir=expected_training_run_dir,
            expected_name='finetuned.ptnn',
            expected_type=Cdr3BertFineTuneWrapper,
            expected_params_template=expected_finetune_parameters_template
        )
    

    def test_finetune_loop_distributed(
        self,
        tmp_finetune_working_dir,
        finetune_hyperparams_path,
        expected_finetune_hyperparams,
        expected_finetune_log_cols,
        expected_finetune_parameters_template
    ):
        n_gpus = torch.cuda.device_count()
        if n_gpus <= 1:
            warn(
                'Finetune distributed test skipped due to hardware '
                'limitations.'
            )
            return

        # Run the training loop in a separate process
        p = mp.Process(
            target=main_f,
            kwargs={
                'working_directory': tmp_finetune_working_dir,
                'run_id': 'test',
                'hyperparams_path': finetune_hyperparams_path,
                'n_gpus': n_gpus,
                'test_mode': True
            }
        )
        p.start()
        p.join()

        expected_training_run_dir = tmp_finetune_working_dir / \
            'finetune_runs' / 'test'

        assert expected_training_run_dir.is_dir()

        check_hyperparam_record(
            training_run_dir=expected_training_run_dir,
            expected_hyperparams=expected_finetune_hyperparams
        )
        for i in range(n_gpus):
            check_training_log(
                training_run_dir=expected_training_run_dir,
                expected_name=f'training_log_cuda_{i}.csv',
                expected_columns=expected_finetune_log_cols,
                expected_length=3
            )
        check_saved_model(
            training_run_dir=expected_training_run_dir,
            expected_name='finetuned_cuda_0.ptnn',
            expected_type=Cdr3BertFineTuneWrapper,
            expected_params_template=expected_finetune_parameters_template
        )
        check_models_equivalent(
            training_run_dir=expected_training_run_dir,
            n_gpus=n_gpus
        )