from src.pipelines import ACLPipeline, ECLPipeline
import multiprocessing as mp
from pathlib import Path
import pytest
from tests.resources.helper_functions import *
import torch
from warnings import warn


def get_config(tmp_path: Path, model_name: str, tokeniser: str, data_file: str) -> dict:
    config = {
        "model": {
            "class": model_name,
            "config": {
                "name": "foobar",
                "num_encoder_layers": 2,
                "d_model": 4,
                "nhead": 2,
                "dim_feedforward": 16,
            },
            "pretrain_state_dict_path": str(tmp_path / "state_dict.pt"),
        },
        "data": {
            "train_path": f"tests/resources/{data_file}",
            "train_background_path": f"tests/resources/{data_file}",
            "valid_path": f"tests/resources/{data_file}",
            "tokeniser": tokeniser,
            "dataset": {"config": {"censoring_lhs": True, "censoring_rhs": True}},
            "dataloader": {"config": {"batch_size": torch.cuda.device_count()}},
        },
        "optim": {
            "contrastive_loss": {"class": "SimCLoss", "config": {"temp": 0.05}},
            "optimiser": {"config": {"n_warmup_steps": 10000}},
        },
        "n_epochs": 3,
        "n_gpus": torch.cuda.device_count(),
    }
    return config


@pytest.mark.parametrize(
    ("pipeline", "model_class", "tokeniser", "data_file"),
    (
        (
            ACLPipeline(),
            "CDR3ClsBERT",
            {"class": "CDR3Tokeniser", "config": {}},
            "mock_data.csv",
        ),
        (
            ECLPipeline(),
            "CDR3ClsBERT",
            {"class": "CDR3Tokeniser", "config": {}},
            "mock_data.csv"
        )
    ),
)
def test_training_loop(
    tmp_path,
    pipeline,
    model_class,
    tokeniser,
    data_file,
):
    if not torch.cuda.is_available():
        warn("ACL test skipped due to hardware limitations.")
        return

    # Set up config
    config = get_config(tmp_path, model_class, tokeniser, data_file)

    # Get the correct model template
    model_template = get_model_template(model_class)

    # Copy toy state_dict into tmp_path
    torch.save(model_template.state_dict(), tmp_path / "state_dict.pt")

    # Run MLM training loop in separate process
    p = mp.Process(
        target=pipeline.main,
        kwargs={"wd": tmp_path, "name": "test", "config": config},
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
            "valid_cont_loss",
            "valid_mlm_loss",
            "valid_aln",
            "valid_unf",
            "valid_mlm_acc",
        ],
        expected_len=4,
    )

    # Check that config json is saved correctly
    assert config_saved(
        save_path=expected_save_dir / "config.json", config_template=config
    )
