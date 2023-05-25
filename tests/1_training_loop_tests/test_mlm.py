from src.pipelines import MLMPipeline
import multiprocessing as mp
import pytest
from tests.resources.helper_functions import *
import torch
from warnings import warn


def get_config(model_class: str, tokeniser: str, data_file: str) -> dict:
    config = {
        "model": {
            "class": model_class,
            "config": {
                "name": "foo (bar, baz 0.1)",
                "num_encoder_layers": 2,
                "d_model": 4,
                "nhead": 2,
                "dim_feedforward": 16,
            },
        },
        "data": {
            "train_path": f"tests/resources/{data_file}",
            "valid_path": f"tests/resources/{data_file}",
            "tokeniser": tokeniser,
            "dataloader": {"config": {"batch_size": torch.cuda.device_count()}},
        },
        "optim": {
            "optimiser": {"config": {"n_warmup_steps": 10000}},
        },
        "n_epochs": 3,
        "n_gpu": torch.cuda.device_count(),
    }
    return config


class TestTrainingLoop:
    @pytest.mark.parametrize(
        ("model_class", "tokeniser", "data_file"),
        (
            (
                "BCDR3BERT",
                {"class": "CDR3Tokeniser", "config": {}},
                "mock_data.csv",
            ),
            (
                "BCDR3BERT",
                {"class": "CDR3Tokeniser", "config": {}},
                "mock_data.csv",
            ),
            (
                "BVCDR3BERT",
                {"class": "BVCDR3Tokeniser", "config": {}},
                "mock_data_beta.csv",
            ),
            (
                "CDRBERT",
                {
                    "class": "CDRTokeniser",
                    "config": {"p_drop_aa": 0, "p_drop_cdr": 0, "p_drop_chain": 0},
                },
                "mock_data.csv",
            ),
        ),
    )
    def test_training_loop(
        self,
        tmp_path,
        model_class,
        tokeniser,
        data_file,
    ):
        if not torch.cuda.is_available():
            warn("MLM test skipped due to hardware limitations.")
            return

        # Set up config
        config = get_config(model_class, tokeniser, data_file)

        # Get the correct model template
        model_template = get_model_template(model_class)

        # Run MLM training loop in separate process
        p = mp.Process(
            target=MLMPipeline.main,
            kwargs={"wd": tmp_path, "config": config},
        )
        p.start()
        p.join()

        expected_save_dir = tmp_path / "model_saves" / "foo_bar_baz_01"
        assert expected_save_dir.is_dir()

        # Check that model is saved correctly
        assert model_saved(
            save_path=expected_save_dir / "state_dict.pt", model_template=model_template
        )

        # Check that log is saved correctly
        assert log_saved(
            save_path=expected_save_dir / "log.csv",
            expected_cols=[
                "epoch",
                "loss",
                "lr",
                "valid_loss",
                "valid_acc",
                "valid_top5_acc",
            ],
            expected_len=4,
        )

        # Check that config json is saved correctly
        assert config_saved(
            save_path=expected_save_dir / "config.json", config_template=config
        )
