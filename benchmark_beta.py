"""
Benchmark models using beta-chain only data.
"""


import argparse
import json
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from scipy.spatial.distance import squareform
import seaborn
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_curve
from src.model_loader import ModelLoader
import torch
from torch import Tensor
from typing import Dict, Tuple

seaborn.set_theme()
seaborn.set_style("white")


PROJECT_DIR = Path(__file__).parent.resolve()
TANNO_DATA_PATH = "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/tanno/test.csv"
VDJDB_DATA_PATH = "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/vdjdb/evaluation_beta.csv"
DASH_DATA_PATH = "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/tcrdist/dash_human.csv"


class BenchmarkingPipeline:
    def __init__(self) -> None:
        tanno_data = pd.read_csv(TANNO_DATA_PATH)
        vdjdb_data = pd.read_csv(VDJDB_DATA_PATH)
        dash_data = pd.read_csv(DASH_DATA_PATH)

        tanno_data[["TRAV", "CDR3A", "TRAJ"]] = pd.NA

        vdjdb_data[["TRAV", "CDR3A", "TRAJ"]] = pd.NA

        dash_data = dash_data[["v_b_gene", "cdr3_b_aa", "epitope", "count"]]
        dash_data = dash_data.rename(
            columns={
                "v_b_gene": "TRBV",
                "cdr3_b_aa": "CDR3B",
                "epitope": "Epitope",
                "count": "duplicate_count",
            }
        )
        dash_data = dash_data.drop_duplicates(
            subset=["TRBV", "CDR3B"], ignore_index=True
        )

        self.bg_data = tanno_data
        self.ep_data = {"vdjdb": vdjdb_data, "dash": dash_data}

    def run_from_clargs(self) -> None:
        parser = argparse.ArgumentParser(description="Benchmarking pipeline.")
        parser.add_argument("model_save_dir", help="Path to model save directory.")
        args = parser.parse_args()

        self.main(model_save_dir=Path(args.model_save_dir))

    def main(self, model_save_dir: Path) -> None:
        print("Setting up...")
        self.setup(model_save_dir)

        summary_dict = {
            "model_name": self.model.name,
        }
        data = dict()
        plots = dict()

        # Benchmarking on background data
        print("Exploring embedding space using background data...")
        pca_summary, pca_projection = self.explore_embspace()
        plots["pca_summary"] = pca_summary
        plots["pca_projection"] = pca_projection

        # Benchmarking on epitope data
        print("Benchmarking on Epitope-labelled data...")
        for ds_name, ds_df in self.ep_data.items():
            embs = self.get_embs(ds_name, ds_df)
            epitopes = torch.tensor(ds_df["Epitope"].astype("category").cat.codes).to(
                self.device
            )

            knn_scores = BenchmarkingPipeline.knn(embs, epitopes)
            avg_precision, precisions, recalls = BenchmarkingPipeline.precision_recall(
                embs, epitopes
            )

            fig, ax = plt.subplots()
            ax.step(recalls, precisions)
            ax.set_title(f"{ds_name} pr curve ({self.model.name})")
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

    def setup(self, model_save_dir: Path) -> None:
        self.model = ModelLoader(model_save_dir)
        self.device = self.model._device

        self.benchmark_dir = PROJECT_DIR / "benchmarks_beta" / self.model.name
        self.cache_dir = self.benchmark_dir / ".cache"

        if not self.benchmark_dir.is_dir():
            self.benchmark_dir.mkdir()
        if not self.cache_dir.is_dir():
            self.cache_dir.mkdir()

    def explore_embspace(self) -> Tuple[Figure]:
        embs = self.get_embs("tanno", self.bg_data)

        pca = PCA()
        pca.fit(embs.cpu())

        # Summary figure
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        summary, summary_ax_1 = plt.subplots()
        summary_ax_1.set_title(f"PCA Summary ({self.model.name})")
        summary_ax_1.set_ylabel("Variance per PCA (bars)")

        summary_ax_2 = summary_ax_1.twinx()
        summary_ax_2.set_ylabel("Cumulative Variance (line)")
        summary_ax_2.set_ylim(0, 1.05)

        summary_ax_1.bar(range(pca.n_components_), pca.explained_variance_ratio_)
        summary_ax_2.plot(cumulative_variance, c="C1")

        # Projection figure
        bg_proj = pca.transform(embs.cpu())[:10000, :2]
        projection = seaborn.jointplot(x=bg_proj[:, 0], y=bg_proj[:, 1])
        projection.set_axis_labels(xlabel="PCA 1", ylabel="PCA 2")

        return summary, projection

    def get_embs(self, ds_name: str, ds_df: DataFrame) -> Tensor:
        save_path = self.cache_dir / f"{ds_name}_embs.pt"

        if save_path.is_file():
            return torch.load(save_path).to(self.device)

        embs = torch.tensor(self.model.embed(ds_df))
        torch.save(embs, save_path)
        return embs.to(self.device)

    @staticmethod
    def knn(embs: Tensor, epitopes: Tensor) -> Dict[str, float]:
        cdist = torch.cdist(embs, embs, p=2)
        cdist.fill_diagonal_(torch.inf)

        knn_scores = dict()

        for k in (5, 10, 50, 100):
            scores = []
            size = len(cdist)

            for tcr_index in range(size):
                expected = epitopes[tcr_index]  # Get correct epitope label
                dists = cdist[tcr_index]  # Get list of distances for that TCR
                _, idcs = torch.topk(
                    dists, k=k, largest=False
                )  # Get indices of nearest neighbours
                neighbouring_epitopes = epitopes[
                    idcs
                ]  # Get epitopes of nearest neighbours
                pred, _ = torch.mode(neighbouring_epitopes)  # Predict epitope
                scores.append(expected.item() == pred.item())  # Record score

            score = torch.tensor(scores, dtype=torch.float32).mean().item()
            knn_scores[k] = score

        return knn_scores

    @staticmethod
    def precision_recall(embs: Tensor, epitopes: Tensor) -> Tuple[float, ndarray]:
        pdist = torch.pdist(embs, p=2)
        probs = torch.exp(-pdist).cpu()
        positive_pair = (epitopes[:, None] == epitopes[None, :]).cpu() & (
            torch.eye(len(epitopes)) != 1
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
    pipeline.run_from_clargs()
