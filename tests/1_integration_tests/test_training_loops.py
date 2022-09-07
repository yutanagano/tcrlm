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


# --- FIXTURES ---


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
        aa_vocab_size=expected_pretrain_hyperparams['aa_vocab_size'],
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

    bert_save_dest = test_dir / 'bert_state_dict.pt'
    generator_save_dest = test_dir / 'generator_state_dict.pt'

    copy(pretrain_hyperparams_path, test_dir / 'hyperparams.csv')

    torch.save(
        pretrained_model_template.bert.state_dict(),
        bert_save_dest
    )
    torch.save(
        pretrained_model_template.generator.state_dict(),
        generator_save_dest
    )

    return tmp_path


# --- HELPER FUNCTIONS ---


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
    model_template: torch.nn.Module,
    device_suffix: str
):
    def state_dicts_same_shape(result, expected):
        if len(result) != len(expected):
            print('state_dicts have different sizes.')
            return False
        
        for key in expected:
            if expected[key].size() != result[key].size():
                print(f'{key} has tensors of different sizes in state_dicts.')
                return False

        return True

    if type(model_template) == models.Cdr3BertPretrainWrapper:
        expected_bert_save_path = \
            training_run_dir / f'bert_state_dict_{device_suffix}.pt'
        expected_generator_save_path = \
            training_run_dir / f'generator_state_dict_{device_suffix}.pt'

        assert expected_bert_save_path.is_file()
        assert expected_generator_save_path.is_file()

        bert_save = torch.load(expected_bert_save_path)
        generator_save = torch.load(expected_generator_save_path)

        assert state_dicts_same_shape(
            result=bert_save,
            expected=model_template.bert.state_dict()
        )
        assert state_dicts_same_shape(
            result=generator_save,
            expected=model_template.generator.state_dict()
        )
        return

    
    if type(model_template) == models.Cdr3BertFineTuneWrapper:
        expected_alpha_bert_save_path = \
            training_run_dir / f'alpha_bert_state_dict_{device_suffix}.pt'
        expected_beta_bert_save_path = \
            training_run_dir / f'beta_bert_state_dict_{device_suffix}.pt'
        expected_classifier_save_path = \
            training_run_dir / f'classifier_state_dict_{device_suffix}.pt'

        assert expected_alpha_bert_save_path.is_file()
        assert expected_beta_bert_save_path.is_file()
        assert expected_classifier_save_path.is_file()

        alpha_bert_save = torch.load(expected_alpha_bert_save_path)
        beta_bert_save = torch.load(expected_beta_bert_save_path)
        classifier_save = torch.load(expected_classifier_save_path)

        assert state_dicts_same_shape(
            result=alpha_bert_save,
            expected=model_template.embedder.alpha_bert.state_dict()
        )
        assert state_dicts_same_shape(
            result=beta_bert_save,
            expected=model_template.embedder.beta_bert.state_dict()
        )
        assert state_dicts_same_shape(
            result=classifier_save,
            expected=model_template.classifier.state_dict()
        )
        return

    raise RuntimeError(
        f'Model template of unknown type: {type(model_template)}.'
    )


def check_state_dicts_equivalent(
    training_run_dir: Path,
    file_base_name: str,
    n_gpus: int
) -> None:
    paths_to_models = [
        training_run_dir / f'{file_base_name}_cuda_{i}.pt' \
        for i in range(n_gpus)
    ]

    for p in paths_to_models:
        assert p.is_file()

    state_dicts = [torch.load(p) for p in paths_to_models]

    def state_dicts_equivalent(sd1, sd2):
        if len(sd1) != len(sd2):
            print('state_dicts have different sizes.')
            return False
    
        for key in sd1:
            if not sd1[key].equal(sd2[key]):
                print(f'{key} has nonequal tensors in state_dicts')
                return False

        return True
        
    for sd2 in state_dicts[1:]:
        assert state_dicts_equivalent(state_dicts[0], sd2)


# --- TESTS ---


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
            model_template=pretrained_model_template,
            device_suffix=('cuda_0' if n_gpus == 1 else 'cpu')
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
            model_template=pretrained_model_template,
            device_suffix='cuda_0'
        )
        check_state_dicts_equivalent(
            training_run_dir=expected_training_run_dir,
            file_base_name='bert_state_dict',
            n_gpus=n_gpus
        )
        check_state_dicts_equivalent(
            training_run_dir=expected_training_run_dir,
            file_base_name='generator_state_dict',
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
            model_template=finetuned_model_template,
            device_suffix=('cuda_0' if n_gpus == 1 else 'cpu')
        )
    

    def test_finetune_loop_distributed(
        self,
        tmp_finetune_working_dir,
        finetune_hyperparams_path,
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
            model_template=finetuned_model_template,
            device_suffix='cuda_0'
        )
        check_state_dicts_equivalent(
            training_run_dir=expected_training_run_dir,
            file_base_name='alpha_bert_state_dict',
            n_gpus=n_gpus
        )
        check_state_dicts_equivalent(
            training_run_dir=expected_training_run_dir,
            file_base_name='beta_bert_state_dict',
            n_gpus=n_gpus
        )
        check_state_dicts_equivalent(
            training_run_dir=expected_training_run_dir,
            file_base_name='classifier_state_dict',
            n_gpus=n_gpus
        )