"""
Benchmark models using beta-chain only data.
"""


from argparse import ArgumentParser
from pathlib import Path
from src.pipelines import BetaBenchmarkingPipeline


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmarking pipeline.")
    parser.add_argument("model_save_dir", help="Path to model save directory.")
    args = parser.parse_args()

    BetaBenchmarkingPipeline.main(Path(args.model_save_dir))
