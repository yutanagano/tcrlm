import json
from autocontrastive import main
import multiprocessing as mp
import pandas as pd
from pathlib import Path
import pytest
from src.modules import CDR3ClsBERT_apc, CDR3BERT_ap
import torch
from torch.nn import Module
from warnings import warn


@pytest.fixture
def cdr3clsbert_apc_template():
    model = CDR3ClsBERT_apc(
        num_encoder_layers=2,
        d_model=4,
        nhead=2,
        dim_feedforward=16
    )
    return model


@pytest.fixture
def cdr3bert_ap_template():
    model = CDR3BERT_ap(
        num_encoder_layers=2,
        d_model=4,
        nhead=2,
        dim_feedforward=16
    )
    return model


def get_config(
    tmp_path: Path,
    model_name: str,
    tokeniser: str,
    data_file: str,
    gpu: bool
) -> dict:
    config = {
        'model': {
            'name': model_name,
            'config': {
                'num_encoder_layers': 2,
                'd_model': 4,
                'nhead': 2,
                'dim_feedforward': 16
            },
            'pretrain_state_dict_path': str(tmp_path/'state_dict.pt')
        },
        'data': {
            'train_path': f'tests/resources/{data_file}',
            'valid_path': f'tests/resources/{data_file}',
            'tokeniser': tokeniser,
            'dataloader_config': {}
        },
        'optim': {
            'contrastive_loss': {
                'name': 'SimCLoss',
                'config': {'temp': 0.05}
            },
            'optimiser_config': {'n_warmup_steps': 10000}
        },
        'n_epochs': 3,
        'gpu': gpu
    }
    return config


def model_saved(save_path: Path, model_template: Module) -> bool:
    result = torch.load(save_path)
    expected = model_template.state_dict()

    if len(result) != len(expected):
        print('state_dicts have different sizes.')
        return False
    
    for key in expected:
        if expected[key].size() != result[key].size():
            print(f'{key} has tensors of different sizes in state_dicts.')
            return False

    return True


def log_saved(save_path: Path, expected_cols: list, expected_len: int) -> bool:
    result = pd.read_csv(save_path)

    if result.columns.to_list() != expected_cols:
        return False

    if len(result) != expected_len:
        return False

    return True


def config_saved(save_path: Path, config_template: dict) -> bool:
    with open(save_path, 'r') as f:
        result = json.load(f)
    
    return result == config_template


class TestTrainingLoop:
    @pytest.mark.parametrize(
        ('model_name', 'tokeniser', 'data_file', 'gpu'),
        (
            ('CDR3ClsBERT_apc', 'ABCDR3Tokeniser', 'mock_data.csv', False),
            ('CDR3ClsBERT_apc', 'ABCDR3Tokeniser', 'mock_data.csv', True),
            ('CDR3BERT_ap', 'BCDR3Tokeniser', 'mock_data_beta.csv', False)
        )
    )
    def test_training_loop(
        self,
        cdr3clsbert_apc_template,
        cdr3bert_ap_template,
        tmp_path,
        model_name,
        tokeniser,
        data_file,
        gpu
    ):
        if gpu and not torch.cuda.is_available():
            warn(
                'Autocontrastive GPU test skipped due to hardware limitations.'
            )
            return

        # Set up config
        config = get_config(tmp_path, model_name, tokeniser, data_file, gpu)

        # Get the correct model template
        if model_name == 'CDR3ClsBERT_apc':
            model_template = cdr3clsbert_apc_template
        elif model_name == 'CDR3BERT_ap':
            model_template = cdr3bert_ap_template

        # Copy toy state_dict into tmp_path
        torch.save(
            model_template.state_dict(),
            tmp_path/'state_dict.pt'
        )

        # Run MLM training loop in separate process
        p = mp.Process(
            target=main,
            kwargs={
                'wd': tmp_path,
                'name': 'test',
                'config': config
            }
        )
        p.start()
        p.join()

        expected_save_dir = tmp_path/'model_saves'/'test'
        assert expected_save_dir.is_dir()

        # Check that model is saved correctly    
        assert model_saved(
            save_path=expected_save_dir/'state_dict.pt',
            model_template=model_template
        )

        # Check that log is saved correctly
        assert log_saved(
            save_path=expected_save_dir/'log.csv',
            expected_cols=[
                'epoch',
                'loss',
                'lr',
                'valid_cont_loss',
                'valid_mlm_loss',
                'valid_aln',
                'valid_unf',
                'valid_mlm_acc'
            ],
            expected_len=4
        )

        # Check that config json is saved correctly
        assert config_saved(
            save_path=expected_save_dir/'config.json',
            config_template=config
        )