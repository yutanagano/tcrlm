import json
from mlm import main
import multiprocessing as mp
import pandas as pd
from pathlib import Path
import pytest
from src.modules import CDR3BERT_ac
import torch
from torch.nn import Module
from warnings import warn


@pytest.fixture
def cdr3bert_c_template():
    model = CDR3BERT_ac(
        num_encoder_layers=2,
        d_model=4,
        nhead=2,
        dim_feedforward=16
    )
    return model


def get_config(gpu: bool) -> dict:
    config = {
        'model': {
            'name': 'CDR3BERT_ac',
            'config': {
                'num_encoder_layers': 2,
                'd_model': 4,
                'nhead': 2,
                'dim_feedforward': 16
            }
        },
        'data': {
            'train_path': 'tests/resources/mock_data.csv',
            'valid_path': 'tests/resources/mock_data.csv',
            'tokeniser': 'ABCDR3Tokeniser',
            'dataloader': {
                'config': {}
            }
        },
        'optim': {
            'optimiser_config': {'n_warmup_steps': 10000},
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
        'gpu', (False, True)
    )
    def test_training_loop(self, cdr3bert_c_template, tmp_path, gpu):
        if gpu and not torch.cuda.is_available():
            warn('MLM GPU test skipped due to hardware limitations.')
            return

        # Set up config
        config = get_config(gpu)

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