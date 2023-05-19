"""
Benchmark models using beta-chain only data.
"""


from .benchmarking_pipeline import BenchmarkingPipeline
import numpy as np
from pathlib import Path
from pandas import DataFrame
import pandas as pd
from tcrdist.repertoire import TCRrep


class TcrdistBenchmarkingPipeline(BenchmarkingPipeline):
    LABELLED_DATA_PATHS = {
        "vdjdb": "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/vdjdb/evaluation_beta.csv",
        "dash": "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/dash/evaluation.csv",
    }
    CHAINS = ["alpha", "beta"]

    def main(cls) -> None:
        cls.setup(None)
        for ds_name, ds_df in cls.labelled_data.items():
            cls.becnhmark_on_labelled_data(ds_name, ds_df)
        cls.save()
        print("Done!")

    def instantiate_model(cls, model_save_dir: Path) -> None:
        cls.model = type("", (), {"name": "tcrdist"})

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

    def get_cdist_matrix(cls, ds_name: str, ds: DataFrame) -> tuple:
        tr = TCRrep(cell_df=ds, organism="human", chains=cls.CHAINS, deduplicate=False)
        cdist_matrix = tr.pw_beta.astype(np.float32)
        epitope_cat_codes = ds["Epitope"].astype("category").cat.codes.to_numpy()

        return cdist_matrix, epitope_cat_codes


class BTcrdistBenchmarkingPipeline(TcrdistBenchmarkingPipeline):
    LABELLED_DATA_PATHS = {
        "vdjdb": "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/vdjdb/evaluation_beta.csv",
        "dash": "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/dash/evaluation.csv",
        "mira": "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/mira/valid.csv",
    }
    CHAINS = ["beta"]
