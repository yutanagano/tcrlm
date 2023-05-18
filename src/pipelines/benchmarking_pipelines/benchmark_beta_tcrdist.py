"""
Benchmark models using beta-chain only data.
"""


import json
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from scipy.spatial.distance import squareform
from scipy.stats import mode
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve
from tcrdist.repertoire import TCRrep
from typing import Dict, Tuple


class BTcrdistBenchmarkingPipeline:
    LABELLED_DATA_PATHS = {
        "vdjdb": "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/vdjdb/evaluation_beta.csv",
        "dash": "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/dash/evaluation.csv",
        "mira": "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/mira/valid.csv"
    }

    model: object
    labelled_data: Dict[str, DataFrame]
    save_dir: Path

    @classmethod
    def main(cls) -> None:
        cls.setup()

        summary_dict = {"model_name": "tcrdist"}
        data = dict()
        plots = dict()

        # Benchmarking on epitope data
        print("Benchmarking on Epitope-labelled data...")
        for ds_name, ds_df in cls.labelled_data.items():
            cdist = cls.get_tcrdist_cdist_matrix(ds_df)
            epitope_cat_codes = ds_df["Epitope"].astype("category").cat.codes.to_numpy()

            knn_scores = cls.knn(cdist, epitope_cat_codes)
            avg_precision, precisions, recalls = cls.precision_recall(
                cdist, epitope_cat_codes
            )

            summary_dict[ds_name] = {
                "knn_scores": knn_scores,
                "avg_precision": avg_precision,
            }
            data[f"{ds_name}_precisions.npy"] = precisions
            data[f"{ds_name}_recalls.npy"] = recalls
            plots[f"{ds_name}_pr_curve.png"] = cls.generate_pr_curve(recalls, precisions, ds_name)

        cls.save(summary_dict, data, plots)

        print("Done!")

    @classmethod
    def setup(cls) -> None:
        # Set model to an empty object with attribute "name" as "tcrdist"
        # This is done for easier compatibility with the general benchmarking
        # framework
        cls.model = type('',(),{'name':'tcrdist'})

        def load_and_transform_data(path):
            df = pd.read_csv(path)
            df = df.rename(columns={"TRBV": "v_b_gene", "CDR3B": "cdr3_b_aa"})
            df = df[["v_b_gene", "cdr3_b_aa", "Epitope"]]
            df["v_b_gene"] = df["v_b_gene"].map(
                lambda x: x if "*" in x else x + "*01"
            )
            df["count"] = 1
            return df
        
        # Load data to benchmark on
        cls.labelled_data = {
            name: load_and_transform_data(path) for name, path in cls.LABELLED_DATA_PATHS.items()
        }

        # Prepare directory in which to save
        benchmark_dir = Path.cwd() / "benchmarks_beta"
        if not benchmark_dir.is_dir():
            benchmark_dir.mkdir()

        cls.save_dir = benchmark_dir / "tcrdist"
        if not cls.save_dir.is_dir():
            cls.save_dir.mkdir()

        # Use seaborn to set pretty defaults for pyplot
        sns.set_theme()
        sns.set_style("white")

    @staticmethod
    def get_tcrdist_cdist_matrix(df: DataFrame) -> ndarray:
        tr = TCRrep(
            cell_df=df, organism="human", chains=["beta"], deduplicate=False
        )
        cdist = tr.pw_beta.astype(np.float32)
        return cdist

    @staticmethod
    def knn(cdist: ndarray, epitope_cat_codes: ndarray) -> Dict[str, float]:
        cdist_cp = cdist.copy()
        np.fill_diagonal(cdist_cp, np.inf)

        knn_scores = dict()

        for k in (5, 10, 50, 100):
            scores = []
            size = len(cdist_cp)

            for tcr_index in range(size):
                expected = epitope_cat_codes[tcr_index]  # Get correct epitope label
                dists = cdist_cp[tcr_index]  # Get list of distances for that TCR
                idcs = np.argsort(dists)[:k]  # Get indices of nearest neighbours

                neighbouring_epitopes = epitope_cat_codes[
                    idcs
                ]  # Get epitopes of nearest neighbours
                pred, _ = mode(neighbouring_epitopes, keepdims=True)  # Predict epitope
                scores.append(expected.item() == pred.item())  # Record score

            score = np.array(scores, dtype=np.float32).mean().item()
            knn_scores[k] = score

        return knn_scores

    @staticmethod
    def precision_recall(cdist: ndarray, epitope_cat_codes: ndarray) -> Tuple[float, ndarray]:
        pdist = squareform(cdist)
        probs = np.exp(-pdist / 50)
        positive_pair = (epitope_cat_codes[:, None] == epitope_cat_codes[None, :]) & (
            np.eye(len(epitope_cat_codes)) != 1
        )
        positive_pair = squareform(positive_pair)

        precisions, recalls, _ = precision_recall_curve(positive_pair, probs)
        avg_precision = average_precision_score(positive_pair, probs)

        return avg_precision, precisions, recalls
    
    @classmethod
    def generate_pr_curve(cls, recalls: ndarray, precisions: ndarray, ds_name: str) -> Figure:
        fig, ax = plt.subplots()
        ax.step(recalls, precisions)
        ax.set_title(f"{ds_name} pr curve ({cls.model.name})")
        ax.set_ylabel("Precision")
        ax.set_xlabel("Recall")
        return fig

    @classmethod
    def save(
        cls, summary_dict: dict, data: Dict[str, ndarray], plots: Dict[str, Figure]
    ) -> None:
        with open(cls.save_dir / "summary.json", "w") as f:
            json.dump(summary_dict, f, indent=4)

        for filename, array in data.items():
            np.save(cls.save_dir / filename, array)

        for filename, plot in plots.items():
            plot.savefig(cls.save_dir / filename)