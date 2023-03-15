from mlm import main
import multiprocessing as mp
import pytest
from tests.resources.helper_functions import *
import torch
from warnings import warn


def get_config(model_class: str, tokeniser: str, data_file: str, gpu: bool) -> dict:
    config = {
        "model": {
            "class": model_class,
            "config": {
                "name": "foobar",
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
            "dataloader": {"config": {}},
        },
        "optim": {
            "optimiser": {"config": {"n_warmup_steps": 10000}},
        },
        "n_epochs": 3,
        "gpu": gpu,
    }
    return config


class TestTrainingLoop:
    @pytest.mark.parametrize(
        ("model_class", "tokeniser", "data_file", "gpu"),
        (
            (
                "BCDR3BERT",
                {"class": "CDR3Tokeniser", "config": {}},
                "mock_data.csv",
                False,
            ),
            (
                "BCDR3BERT",
                {"class": "CDR3Tokeniser", "config": {}},
                "mock_data.csv",
                True,
            ),
            (
                "BVCDR3BERT",
                {"class": "BVCDR3Tokeniser", "config": {}},
                "mock_data_beta.csv",
                False,
            ),
            (
                "CDRBERT",
                {
                    "class": "CDRTokeniser",
                    "config": {"p_drop_aa": 0, "p_drop_cdr": 0, "p_drop_chain": 0},
                },
                "mock_data.csv",
                False,
            ),
        ),
    )
    def test_training_loop(
        self,
        tmp_path,
        model_class,
        tokeniser,
        data_file,
        gpu,
    ):
        if gpu and not torch.cuda.is_available():
            warn("MLM GPU test skipped due to hardware limitations.")
            return

        # Set up config
        config = get_config(model_class, tokeniser, data_file, gpu)

        # Get the correct model template
        model_template = get_model_template(model_class)

        # Run MLM training loop in separate process
        p = mp.Process(
            target=main, kwargs={"wd": tmp_path, "name": "test", "config": config}
        )
        p.start()
        p.join()

        expected_save_dir = tmp_path / "model_saves" / "test"
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
            expected_len=3,
        )

        # Check that config json is saved correctly
        assert config_saved(
            save_path=expected_save_dir / "config.json", config_template=config
        )
