from acl import main
import multiprocessing as mp
from pathlib import Path
import pytest
from tests.resources.helper_functions import *
import torch
from warnings import warn


def get_config(
    tmp_path: Path,
    model_name: str,
    tokeniser: str,
    data_file: str,
    gpu: bool
) -> dict:
    config = {
        'model': {
            'class': model_name,
            'config': {
                'name': 'foobar',
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


class TestTrainingLoop:
    @pytest.mark.parametrize(
        ('model_name', 'tokeniser', 'data_file', 'gpu'),
        (
            ('CDR3ClsBERT_apc', {'class': 'CDR3Tokeniser', 'config': {}}, 'mock_data.csv', False),
            ('CDR3ClsBERT_apc', {'class': 'CDR3Tokeniser', 'config':{}}, 'mock_data.csv', True),
            ('CDR3BERT_a', {'class': 'BCDR3Tokeniser', 'config': {'p_drop_aa': 0}}, 'mock_data_beta.csv', False)
        )
    )
    def test_training_loop(
        self,
        cdr3clsbert_apc_template,
        cdr3bert_a_template,
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
        elif model_name == 'CDR3BERT_a':
            model_template = cdr3bert_a_template

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