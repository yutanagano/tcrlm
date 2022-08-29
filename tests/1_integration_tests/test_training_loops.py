from copy import deepcopy
from finetune import main as main_f
import multiprocessing as mp
import pandas as pd
from pathlib import Path
from pretrain import main as main_p
import pytest
from shutil import copy
import source.nn.models as models
from source.utils.misc import check_dataframe_format
from source.utils.fileio import parse_hyperparams
import torch
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
def pretrained_model_template(expected_pretrain_hyperparams):
    bert = models.Cdr3Bert(
        num_encoder_layers=expected_pretrain_hyperparams['num_encoder_layers'],
        d_model=expected_pretrain_hyperparams['d_model'],
        nhead=expected_pretrain_hyperparams['nhead'],
        dim_feedforward=expected_pretrain_hyperparams['dim_feedforward'],
        activation=expected_pretrain_hyperparams['activation']
    )
    return models.Cdr3BertPretrainWrapper(bert)


@pytest.fixture(scope='module')
def finetuned_model_template(pretrained_model_template):
    alpha_model = deepcopy(pretrained_model_template.bert)
    beta_model = deepcopy(pretrained_model_template.bert)
    embedder = models.TcrEmbedder(alpha_bert=alpha_model, beta_bert=beta_model)
    return models.Cdr3BertFineTuneWrapper(tcr_embedder=embedder)


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


@pytest.fixture(scope='function')
def tmp_finetune_working_dir(
    tmp_path,
    pretrain_hyperparams_path,
    pretrained_model_template
):
    pretrain_runs_dir = tmp_path / 'pretrain_runs'
    pretrain_runs_dir.mkdir()

    test_dir = pretrain_runs_dir / 'test'
    test_dir.mkdir()

    state_dict_dest = test_dir / 'pretrained_state_dict.pt'

    copy(pretrain_hyperparams_path, test_dir / 'hyperparams.csv')
    torch.save(pretrained_model_template.state_dict(), state_dict_dest)

    return tmp_path


def check_hyperparam_record(
    training_run_dir: Path,
    expected_hyperparams: dict
):
    expected_location = training_run_dir / 'hyperparams.csv'
    assert expected_location.is_file()
    
    result_hyperparams = parse_hyperparams(expected_location)
    assert result_hyperparams == expected_hyperparams


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
    state_dict_template: dict[torch.Tensor]
):
    expected_location = training_run_dir / expected_name
    assert expected_location.is_file()

    saved_state_dict = torch.load(expected_location)
    assert len(saved_state_dict) == len(state_dict_template)
    for key in state_dict_template:
        assert saved_state_dict[key].size() == state_dict_template[key].size()


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
        pretrained_model_template,
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
            expected_name='pretrained_state_dict.pt',
            state_dict_template=pretrained_model_template.state_dict()
        )


    def test_pretrain_loop_distributed(
        self,
        tmp_path,
        pretrain_hyperparams_path,
        expected_pretrain_hyperparams,
        expected_pretrain_log_cols,
        pretrained_model_template,
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
            expected_name='pretrained_state_dict_cuda_0.pt',
            state_dict_template=pretrained_model_template.state_dict()
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
        finetuned_model_template,
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
            expected_name='finetuned_state_dict.pt',
            state_dict_template=finetuned_model_template.state_dict()
        )
    

    def test_finetune_loop_distributed(
        self,
        tmp_finetune_working_dir,
        finetune_hyperparams_path,
        expected_pretrain_hyperparams,
        expected_finetune_hyperparams,
        expected_finetune_log_cols,
        finetuned_model_template,
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
            expected_name='finetuned_state_dict_cuda_0.pt',
            state_dict_template=finetuned_model_template.state_dict()
        )
        check_models_equivalent(
            training_run_dir=expected_training_run_dir,
            n_gpus=n_gpus
        )