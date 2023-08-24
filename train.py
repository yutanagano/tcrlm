import argparse
from argparse import Namespace
import json
from pathlib import Path

from src.model_trainer import ModelTrainer


def main():
    args = parse_args()
    path_to_working_directory = get_path_to_working_directory(args)
    config_dict = load_config_dict(args)

    model_trainer = ModelTrainer(config_dict)
    model_trainer.train(working_directory=path_to_working_directory)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Model training executable script.")
    parser.add_argument(
        "-d",
        "--working-directory",
        help="Path to tcr_embedder project working directory.",
    )
    parser.add_argument(
        "config_path", help="Path to the training run config json file."
    )
    return parser.parse_args()


def get_path_to_working_directory(args: Namespace) -> Path:
    if args.working_directory is None:
        path_to_working_directory = Path.cwd()
    else:
        path_to_working_directory = Path(args.working_directory).resolve(strict=True)

    return path_to_working_directory


def load_config_dict(args: Namespace) -> dict:
    with open(args.config_path, "r") as f:
        config = json.load(f)
    return config


if __name__ == "__main__":
    main()
