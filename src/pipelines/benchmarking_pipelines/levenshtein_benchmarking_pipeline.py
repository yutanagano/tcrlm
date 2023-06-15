"""
Simple levenshtein editions of the benchmarking pipeline.
"""


from numpy import ndarray
from .pure_metric_benchmarking_pipeline import PureMetricBenchmarkingPipeline
from pandas import DataFrame
from rapidfuzz.process import cdist
from rapidfuzz.distance import Levenshtein
import tidytcells as tt


class CDR3BLevenshteinBenchmarkingPipeline(PureMetricBenchmarkingPipeline):
    MODEL_NAME = "cdr3b_levenshtein"

    def get_pdist_matrix(cls, ds_name: str, ds_df: DataFrame) -> ndarray:
        return cls.get_cdist_matrix(ds_df, ds_df)
    
    def get_cdist_matrix(cls, ds_a_df: DataFrame, ds_b_df: DataFrame) -> ndarray:
        cdist_matrix = cdist(
            ds_a_df["CDR3B"], ds_b_df["CDR3B"], scorer=Levenshtein.distance
        ).astype(float)
    
        return cdist_matrix

class CDRBLevenshteinBenchmarkingPipeline(PureMetricBenchmarkingPipeline):
    MODEL_NAME = "cdrb_levenshtein"

    def load_data(cls) -> None:
        super().load_data()
        missing_bv_mask = cls.background_data["TRBV"].notna()
        cls.background_data = cls.background_data[
            missing_bv_mask
        ].reset_index(drop=True)
        cls.background_pgen = cls.background_pgen[
            missing_bv_mask
        ]

    def get_pdist_matrix(cls, ds_name: str, ds_df: DataFrame) -> ndarray:
        return cls.get_cdist_matrix(ds_df, ds_df)
    
    def get_cdist_matrix(cls, ds_a_df: DataFrame, ds_b_df: DataFrame) -> ndarray:
        ds_a_df = ds_a_df.copy()
        ds_b_df = ds_b_df.copy()

        def fix_bv(df):
            df["TRBV"] = df["TRBV"].map(lambda x: x if "*" in x else x + "*01")
            return df

        def get_v_gene_cdrs(df):
            df["CDR1B"] = df["TRBV"].map(lambda x: tt.tcr.get_aa_sequence(x)["CDR1-IMGT"])
            df["CDR2B"] = df["TRBV"].map(lambda x: tt.tcr.get_aa_sequence(x)["CDR2-IMGT"])
            return df

        ds_a_df = fix_bv(ds_a_df)
        ds_b_df = fix_bv(ds_b_df)

        ds_a_df = get_v_gene_cdrs(ds_a_df)
        ds_b_df =get_v_gene_cdrs(ds_b_df)

        cdist_cdr1 = cdist(
            ds_a_df["CDR1B"], ds_b_df["CDR1B"], scorer=Levenshtein.distance
        ).astype(float)
        cdist_cdr2 = cdist(
            ds_a_df["CDR2B"], ds_b_df["CDR2B"], scorer=Levenshtein.distance
        ).astype(float)
        cdist_cdr3 = cdist(
            ds_a_df["CDR3B"], ds_b_df["CDR3B"], scorer=Levenshtein.distance
        ).astype(float)
    
        return cdist_cdr1 + cdist_cdr2 + cdist_cdr3