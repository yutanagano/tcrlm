"""
Simple levenshtein editions of the benchmarking pipeline.
"""


from .pure_metric_benchmarking_pipeline import PureMetricBenchmarkingPipeline
from pandas import DataFrame
from rapidfuzz.process import cdist
from rapidfuzz.distance import Levenshtein


class CDR3BLevenshteinBenchmarkingPipeline(PureMetricBenchmarkingPipeline):
    MODEL_NAME = "cdr3b_levenshtein"

    def get_cdist_matrix(cls, ds_name: str, ds_df: DataFrame) -> tuple:
        cdist_matrix = cdist(
            ds_df["CDR3B"], ds_df["CDR3B"], scorer=Levenshtein.distance
        ).astype(float)

        return cdist_matrix