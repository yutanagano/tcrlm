"""
tcrdist edition of benchmarking pipelines.
"""


from .pure_metric_benchmarking_pipeline import PureMetricBenchmarkingPipeline
import numpy as np
from pathlib import Path
from pandas import DataFrame
import pandas as pd
from tcrdist.repertoire import TCRrep


class TcrdistBenchmarkingPipeline(PureMetricBenchmarkingPipeline):
    MODEL_NAME = "tcrdist"
    CHAINS = ["alpha", "beta"]

    @staticmethod
    def load_csv(path: Path) -> DataFrame:
        df = pd.read_csv(path)
        df = df.rename(
            columns={
                "TRAV": "v_a_gene",
                "CDR3A": "cdr3_a_aa",
                "TRBV": "v_b_gene",
                "CDR3B": "cdr3_b_aa",
                "duplicate_count": "count",
            }
        )
        for column in ["v_a_gene", "v_b_gene"]:
            if column in df:
                df[column] = df[column].map(
                    lambda x: x if not type(x) == str or "*" in x else x + "*01"
                )
        if not "count" in df:
            df["count"] = 1
        return df

    def get_pdist_matrix(cls, ds_name: str, ds_df: DataFrame) -> tuple:
        tr = TCRrep(cell_df=ds_df, organism="human", chains=cls.CHAINS, deduplicate=False)
        pdist_matrix = tr.pw_beta.astype(np.float32)

        return pdist_matrix
    
    def evaluate_pgen_vs_representation_space_density(cls) -> None:
        pass


class BTcrdistBenchmarkingPipeline(TcrdistBenchmarkingPipeline):
    CHAINS = ["beta"]
