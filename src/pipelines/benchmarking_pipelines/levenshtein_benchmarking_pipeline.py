"""
Simple levenshtein editions of the benchmarking pipeline.
"""


from .pure_metric_benchmarking_pipeline import PureMetricBenchmarkingPipeline
from pandas import DataFrame
from rapidfuzz.process import cdist
from rapidfuzz.distance import Levenshtein
import tidytcells as tt


class CDR3BLevenshteinBenchmarkingPipeline(PureMetricBenchmarkingPipeline):
    MODEL_NAME = "cdr3b_levenshtein"

    def get_cdist_matrix(cls, ds_name: str, ds_df: DataFrame) -> tuple:
        cdist_matrix = cdist(
            ds_df["CDR3B"], ds_df["CDR3B"], scorer=Levenshtein.distance
        ).astype(float)

        return cdist_matrix
    

class CDRBLevenshteinBenchmarkingPipeline(PureMetricBenchmarkingPipeline):
    MODEL_NAME = "cdrb_levenshtein"

    def get_cdist_matrix(cls, ds_name: str, ds_df: DataFrame) -> tuple:
        ds_df["TRBV"] = ds_df["TRBV"].map(lambda x: x if "*" in x else x + "*01")

        ds_df["CDR1B"] = ds_df["TRBV"].map(lambda x: tt.tcr.get_aa_sequence(x)["CDR1-IMGT"])
        ds_df["CDR2B"] = ds_df["TRBV"].map(lambda x: tt.tcr.get_aa_sequence(x)["CDR2-IMGT"])

        cdist_cdr1 = cdist(
            ds_df["CDR1B"], ds_df["CDR1B"], scorer=Levenshtein.distance
        ).astype(float)
        cdist_cdr2 = cdist(
            ds_df["CDR2B"], ds_df["CDR2B"], scorer=Levenshtein.distance
        ).astype(float)
        cdist_cdr3 = cdist(
            ds_df["CDR3B"], ds_df["CDR3B"], scorer=Levenshtein.distance
        ).astype(float)

        cdist_matrix = cdist_cdr1 + cdist_cdr2 + cdist_cdr3

        return cdist_matrix