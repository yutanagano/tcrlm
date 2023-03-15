import json
import pandas as pd
from pathlib import Path
import torch
from torch.nn import Module


def model_saved(save_path: Path, model_template: Module) -> bool:
    result = torch.load(save_path)
    expected = model_template.state_dict()

    if len(result) != len(expected):
        print("state_dicts have different sizes.")
        return False

    for key in expected:
        if expected[key].size() != result[key].size():
            print(f"{key} has tensors of different sizes in state_dicts.")
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
    with open(save_path, "r") as f:
        result = json.load(f)

    return result == config_template
