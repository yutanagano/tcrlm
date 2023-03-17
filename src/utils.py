"""
Utility classes and functions.
"""


import json
import pandas as pd
from pathlib import Path
import torch
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel


def save(wd: Path, save_name: str, model: Module, log: dict, config: dict) -> None:
    model_saves_dir = wd / "model_saves"
    try:
        model_saves_dir.mkdir()
    except FileExistsError:
        pass

    try:
        (model_saves_dir / save_name).mkdir()
    except FileExistsError:
        suffix_int = 1
        new_save_name = f"{save_name}_{suffix_int}"
        done = False
        while not done:
            try:
                (model_saves_dir / new_save_name).mkdir()
                save_name = new_save_name
                done = True
            except FileExistsError:
                suffix_int += 1
                new_save_name = f"{save_name}_{suffix_int}"
    save_dir = model_saves_dir / save_name

    # Save model
    model.cpu()
    if isinstance(model, DistributedDataParallel):
        model = model.module
    torch.save(model.state_dict(), save_dir / "state_dict.pt")

    # Save log
    pd.DataFrame.from_dict(log, orient="index").to_csv(
        save_dir / "log.csv", index_label="epoch"
    )

    # Save config
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
