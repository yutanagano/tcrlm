"""
Benchmark models using beta-chain only data.
"""


from pathlib import Path
from pandas import DataFrame
from .benchmarking_pipeline import BenchmarkingPipeline
import pandas as pd


class BetaBenchmarkingPipeline(BenchmarkingPipeline):
    @staticmethod
    def load_csv(path: Path) -> DataFrame:
        df = pd.read_csv(path)
        df[["TRAV", "CDR3A", "TRAJ"]] = pd.NA
        return df
