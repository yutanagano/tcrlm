"""
Benchmark models using beta-chain only data.
"""


import json
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from scipy.spatial.distance import squareform
from scipy.stats import mode
import seaborn
from sklearn.metrics import average_precision_score, precision_recall_curve
from tcrdist.repertoire import TCRrep
from typing import Dict, Tuple

seaborn.set_theme()
seaborn.set_style("white")


PROJECT_DIR = Path(__file__).parent.resolve()
TANNO_DATA_PATH = "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/tanno/test.csv"
VDJDB_DATA_PATH = "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/vdjdb/evaluation_beta.csv"
DASH_DATA_PATH = "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/tcrdist/dash_human.csv"


class BenchmarkingPipeline:
    def __init__(self) -> None:
        vdjdb_data = pd.read_csv(VDJDB_DATA_PATH)
        dash_data = pd.read_csv(DASH_DATA_PATH)

        vdjdb_data = vdjdb_data.rename(
            columns={
                "TRBV": "v_b_gene",
                "CDR3B": "cdr3_b_aa"
            }
        )
        vdjdb_data = vdjdb_data[["v_b_gene", "cdr3_b_aa", "Epitope"]]
        vdjdb_data["count"] = 1

        dash_data = dash_data[["v_b_gene", "cdr3_b_aa", "epitope", "count"]]
        dash_data = dash_data.rename(
            columns={
                "epitope": "Epitope"
            }
        )
        dash_data = dash_data.drop_duplicates(
            subset=["v_b_gene", "cdr3_b_aa"], ignore_index=True
        )
        dash_data["count"] = 1

        self.ep_data = {"vdjdb": vdjdb_data, "dash": dash_data}

        self.benchmark_dir = PROJECT_DIR / "benchmarks_beta" / "tcrdist"

        if not self.benchmark_dir.is_dir():
            self.benchmark_dir.mkdir()

    def main(self) -> None:
        summary_dict = {
            "model_name": "tcrdist"
        }
        data = dict()
        plots = dict()

        # Benchmarking on epitope data
        print("Benchmarking on Epitope-labelled data...")
        for ds_name, ds_df in self.ep_data.items():
            tr = TCRrep(
                cell_df=ds_df,
                organism="human",
                chains=["beta"]
            )
            cdist = tr.pw_beta.astype(np.float32)
            ds_df = tr.clone_df
            epitopes = ds_df["Epitope"].astype("category").cat.codes.to_numpy()

            knn_scores = BenchmarkingPipeline.knn(cdist, epitopes)
            avg_precision, precisions, recalls = BenchmarkingPipeline.precision_recall(cdist, epitopes)

            fig, ax = plt.subplots()
            ax.step(recalls, precisions)
            ax.set_title(f"{ds_name} pr curve (tcrdist)")
            ax.set_ylabel("Precision")
            ax.set_xlabel("Recall")

            summary_dict[ds_name] = {
                "knn_scores": knn_scores,
                "avg_precision": avg_precision,
            }
            data[f"{ds_name}_precisions.npy"] = precisions
            data[f"{ds_name}_recalls.npy"] = recalls
            plots[f"{ds_name}_pr_curve.png"] = fig

        self.save(summary_dict, data, plots)

        print("Done!")

    @staticmethod
    def knn(cdist: ndarray, epitopes: ndarray) -> Dict[str, float]:
        cdist_cp = cdist.copy()
        np.fill_diagonal(cdist_cp, np.inf)

        knn_scores = dict()

        for k in (5, 10, 50, 100):
            scores = []
            size = len(cdist_cp)

            for tcr_index in range(size):
                expected = epitopes[tcr_index]  # Get correct epitope label
                dists = cdist_cp[tcr_index]  # Get list of distances for that TCR
                idcs = np.argsort(dists)[:k]  # Get indices of nearest neighbours

                neighbouring_epitopes = epitopes[
                    idcs
                ]  # Get epitopes of nearest neighbours
                pred, _ = mode(neighbouring_epitopes, keepdims=True)  # Predict epitope
                scores.append(expected.item() == pred.item())  # Record score

            score = np.array(scores, dtype=np.float32).mean().item()
            knn_scores[k] = score

        return knn_scores

    @staticmethod
    def precision_recall(cdist: ndarray, epitopes: ndarray) -> Tuple[float, ndarray]:
        pdist = squareform(cdist)
        probs = np.exp(-pdist/50)
        positive_pair = (epitopes[:, None] == epitopes[None, :]) & (
            np.eye(len(epitopes)) != 1
        )
        positive_pair = squareform(positive_pair)

        precisions, recalls, _ = precision_recall_curve(positive_pair, probs)
        avg_precision = average_precision_score(positive_pair, probs)

        return avg_precision, precisions, recalls

    def save(
        self, summary_dict: dict, data: Dict[str, ndarray], plots: Dict[str, Figure]
    ) -> None:
        with open(self.benchmark_dir / "summary.json", "w") as f:
            json.dump(summary_dict, f, indent=4)

        for filename, array in data.items():
            np.save(self.benchmark_dir / filename, array)

        for filename, plot in plots.items():
            plot.savefig(self.benchmark_dir / filename)


pipeline = BenchmarkingPipeline()


if __name__ == "__main__":
    pipeline.main()
