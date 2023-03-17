import json
import numpy as np
import pandas as pd
import pytest
from src.datahandling.tokenisers import CDR3Tokeniser
from src.models import RandomEmbedder
from src import utils
import torch
from torch.nn import Linear


@pytest.fixture
def toy_model():
    return Linear(3, 3)


@pytest.fixture
def toy_log():
    return {0: {"loss": 2, "valid_loss": 2}, 1: {"loss": 1, "valid_loss": 1}}


@pytest.fixture
def toy_log_df(toy_log):
    df = pd.DataFrame.from_dict(toy_log, orient="index")
    df = df.reset_index().rename(columns={"index": "epoch"})
    return df


@pytest.fixture
def toy_config():
    return {"model": "Linear", "n_epochs": 1337}


def state_dicts_equivalent(state_dict_1: dict, state_dict_2: dict) -> bool:
    if len(state_dict_1) != len(state_dict_2):
        return False

    for key in state_dict_1:
        if not state_dict_1[key].equal(state_dict_2[key]):
            return False

    return True


def is_on_cpu(state_dict: dict) -> bool:
    for key in state_dict:
        if state_dict[key].device.type != "cpu":
            return False

    return True


class TestSave:
    def test_save(self, toy_model, toy_log, toy_log_df, toy_config, tmp_path):
        model_saves_dir = tmp_path / "model_saves"
        test_dir = model_saves_dir / "test"

        utils.save(
            wd=tmp_path,
            save_name="test",
            model=toy_model,
            log=toy_log,
            config=toy_config,
        )

        saved_sd = torch.load(test_dir / "state_dict.pt")

        assert model_saves_dir.is_dir()
        assert test_dir.is_dir()
        assert is_on_cpu(saved_sd)
        assert state_dicts_equivalent(saved_sd, toy_model.state_dict())
        assert pd.read_csv(test_dir / "log.csv").equals(toy_log_df)
        with open(test_dir / "config.json", "r") as f:
            assert json.load(f) == toy_config

    def test_existing_model_saves_dir(
        self, toy_model, toy_log, toy_log_df, toy_config, tmp_path
    ):
        model_saves_dir = tmp_path / "model_saves"
        model_saves_dir.mkdir()
        test_dir = model_saves_dir / "test"

        utils.save(
            wd=tmp_path,
            save_name="test",
            model=toy_model,
            log=toy_log,
            config=toy_config,
        )

        saved_sd = torch.load(test_dir / "state_dict.pt")

        assert model_saves_dir.is_dir()
        assert test_dir.is_dir()
        assert is_on_cpu(saved_sd)
        assert state_dicts_equivalent(saved_sd, toy_model.state_dict())
        assert pd.read_csv(test_dir / "log.csv").equals(toy_log_df)
        with open(test_dir / "config.json", "r") as f:
            assert json.load(f) == toy_config

    def test_save_name_collision(
        self, toy_model, toy_log, toy_log_df, toy_config, tmp_path
    ):
        model_saves_dir = tmp_path / "model_saves"
        model_saves_dir.mkdir()
        (model_saves_dir / "test").mkdir()
        (model_saves_dir / "test_1").mkdir()
        test_dir = model_saves_dir / "test_2"

        utils.save(
            wd=tmp_path,
            save_name="test",
            model=toy_model,
            log=toy_log,
            config=toy_config,
        )

        saved_sd = torch.load(test_dir / "state_dict.pt")

        assert model_saves_dir.is_dir()
        assert test_dir.is_dir()
        assert is_on_cpu(saved_sd)
        assert state_dicts_equivalent(saved_sd, toy_model.state_dict())
        assert pd.read_csv(test_dir / "log.csv").equals(toy_log_df)
        with open(test_dir / "config.json", "r") as f:
            assert json.load(f) == toy_config
