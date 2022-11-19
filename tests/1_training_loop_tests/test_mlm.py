import json
from mlm import main
import multiprocessing as mp
import pandas as pd
from pathlib import Path
import pytest
from src.modules import CDR3BERT_c
import torch
from torch.nn import Module
from warnings import warn


mp.set_start_method('spawn')


@pytest.fixture
def cdr3bert_c_template():
    model = CDR3BERT_c(
        num_encoder_layers=2,
        d_model=4,
        nhead=2,
        dim_feedforward=16
    )
    return model


def get_config(n_gpus: int) -> dict:
    config = {
        'model': 'CDR3BERT_c',
        'model_config': {
            'num_encoder_layers': 2,
            'd_model': 4,
            'nhead': 2,
            'dim_feedforward': 16
        },
        'train_data_path': 'tests/resources/mock_data.csv',
        'valid_data_path': 'tests/resources/mock_data.csv',
        'tokeniser': 'CDR3Tokeniser',
        'dataloader_config': {},
        'optimiser_config': {'n_warmup_steps': 10000},
        'n_epochs': 3,
        'n_gpus': n_gpus
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


class TestMLM:
    @pytest.mark.parametrize(
        'n_gpus', (0, 1, 2)
    )
    def test_mlm(self, cdr3bert_c_template, tmp_path, n_gpus):
        if n_gpus == 1 and not torch.cuda.is_available():
            warn('MLM GPU test skipped due to hardware limitations.')
            return
        
        if n_gpus == 2 and torch.cuda.device_count() < 2:
            warn('MLM distributed test skipped due to hardware limitations.')
            return

        # Set up config
        config = get_config(n_gpus)

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
            model_template=cdr3bert_c_template
        )

        # Check that log is saved correctly
        assert log_saved(
            save_path=expected_save_dir/'log.csv',
            expected_cols=[
                'epoch',
                'loss',
                'lr',
                'valid_loss',
                'valid_acc',
                'valid_top5_acc'
            ],
            expected_len=3
        )

        # Check that config json is saved correctly
        assert config_saved(
            save_path=expected_save_dir/'config.json',
            config_template=config
        )