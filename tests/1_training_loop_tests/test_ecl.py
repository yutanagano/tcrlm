from ecl import main
import multiprocessing as mp
from pathlib import Path
import pytest
from tests.resources.helper_functions import *
import torch
from warnings import warn


def get_config(tmp_path: Path, gpu: bool) -> dict:
    config = {
        'model': {
            'class': 'CDR3ClsBERT_apc',
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
            'train_path': {
                'autocontrastive': 'tests/resources/mock_data.csv',
                'epitope_contrastive': 'tests/resources/mock_data.csv'
            },
            'valid_path': {
                'autocontrastive': 'tests/resources/mock_data.csv',
                'epitope_contrastive': 'tests/resources/mock_data.csv'
            },
            'tokeniser': 'CDR3Tokeniser',
            'autocontrastive_noising': False,
            'dataloader_config': {}
        },
        'optim': {
            'autocontrastive_loss': {
                'name': 'SimCLoss',
                'config': {'temp': 0.05}
            },
            'epitope_contrastive_loss': {
                'name': 'PosBackSimCLoss',
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
        'gpu', (False, True)
    )
    def test_training_loop(
        self,
        cdr3clsbert_apc_template,
        tmp_path,
        gpu
    ):
        if gpu and not torch.cuda.is_available():
            warn(
                'Epitope contrastive GPU test '
                'skipped due to hardware limitations.'
            )
            return

        # Set up config
        config = get_config(tmp_path, gpu)

        # Copy toy state_dict into tmp_path
        torch.save(
            cdr3clsbert_apc_template.state_dict(),
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
            model_template=cdr3clsbert_apc_template
        )

        # Check that log is saved correctly
        assert log_saved(
            save_path=expected_save_dir/'log.csv',
            expected_cols=[
                'epoch',
                'loss',
                'lr',
                'valid_ec_loss',
                'valid_ac_loss',
                'valid_mlm_loss',
                'valid_epitope_aln',
                'valid_auto_aln',
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