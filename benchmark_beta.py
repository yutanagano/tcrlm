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
import re
from scipy.spatial.distance import squareform
import seaborn
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_curve
from src.datahandling.datasets import TCRDataset
from src.datahandling.dataloaders import MLMDataLoader
from src.metrics import mlm_acc
from src.model_loader import ModelLoader
from src.resources import AMINO_ACIDS
import torch
from torch import topk
from torch.nn.functional import softmax
from torch import Tensor
from tqdm import tqdm
from typing import Dict, Tuple

seaborn.set_theme()
seaborn.set_style("white")


PROJECT_DIR = Path(__file__).parent.resolve()
TANNO_DATA_PATH = "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/tanno/test.csv"
VDJDB_DATA_PATH = "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/vdjdb/evaluation_beta.csv"
DASH_DATA_PATH = "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/tcrdist/dash_human.csv"
MIRA_DATA_PATH = "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/mira/valid.csv"


class BenchmarkingPipeline:
    model: ModelLoader
    device: torch.device
    benchmark_dir: Path
    cache_dir: Path

    def __init__(self) -> None:
        tanno_data = pd.read_csv(TANNO_DATA_PATH)
        vdjdb_data = pd.read_csv(VDJDB_DATA_PATH)
        dash_data = pd.read_csv(DASH_DATA_PATH)
        mira_data = pd.read_csv(MIRA_DATA_PATH)

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

        self.bg_data = tanno_data
        self.ep_data = {"vdjdb": vdjdb_data, "dash": dash_data, "mira": mira_data}

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

        # Benchmarking MLM performance
        print("Benchmarking MLM performance...")
        mlm_summary, mlm_figures = self.benchmark_mlm()
        summary_dict["mlm_summary"] = mlm_summary
        plots = {**plots, **mlm_figures}

        # Benchmarking on background data
        print("Exploring embedding space using background data...")
        pca_summary, pca_projection = self.explore_embspace()
        plots["pca_summary"] = pca_summary
        plots["pca_projection"] = pca_projection

        # Benchmarking on epitope data
        print("Benchmarking on Epitope-labelled data...")
        for ds_name, ds_df in self.ep_data.items():
            (
                performance_summary,
                precisions,
                recalls,
                pr_curve,
            ) = self.benchmark_epitopes(ds_name, ds_df)

            summary_dict[ds_name] = performance_summary
            data[f"{ds_name}_precisions.npy"] = precisions
            data[f"{ds_name}_recalls.npy"] = recalls
            plots[f"{ds_name}_pr_curve.png"] = pr_curve

        self.save(summary_dict, data, plots)

        print("Done!")

    def benchmark_mlm(self) -> tuple:
        model_class_name = self.model.model.__class__.__name__
        dataset = TCRDataset(data=self.bg_data, tokeniser=self.model._tokeniser)
        dataloader = MLMDataLoader(
            dataset=dataset,
            batch_size=512,
            shuffle=False,
            p_mask=0.01,
            p_mask_random=0,
            p_mask_keep=0,
        )

        if re.search("CDR3(Cls)?BERT", model_class_name):
            # Quantify MLM performance
            total_acc = 0
            divisor = 0

            for x, y in tqdm(dataloader):
                num_samples = len(x)

                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model.model.mlm(x)

                total_acc += mlm_acc(logits, y) * num_samples
                divisor += num_samples

            mlm_summary = {"acc": total_acc / divisor}

            # Generate exemplar plots
            exemplars = self.bg_data.sample(5, random_state=420, ignore_index=True)
            exemplar_figures = dict()

            for row_idx, exemplar in exemplars.iterrows():
                # Tokenise
                tokenised = self.model._tokeniser.tokenise(exemplar).to(self.device)

                # Mask each residue and note mlm predictions
                preds_collection = dict()
                for residue_idx in range(1, len(tokenised)):
                    masked = tokenised.detach().clone()
                    masked[residue_idx, 0] = 0
                    logits = self.model.model.mlm(masked.unsqueeze(0))[0, residue_idx]
                    scores = softmax(logits, dim=0)
                    pred_scores, preds_indices = topk(scores, 5, dim=0)
                    preds = [AMINO_ACIDS[idx] for idx in preds_indices]
                    preds_collection[residue_idx] = (preds, pred_scores.detach().cpu())

                # Visualise as a figure
                figure = plt.figure(figsize=(len(tokenised) - 1, 5))
                # For each token in sequence
                for i, token in enumerate(tokenised[1:]):
                    # Write out the correct token
                    symbol = figure.add_subplot(4, len(tokenised) - 1, i + 1)
                    symbol.axis("off")
                    symbol.text(
                        0.5,
                        0,
                        AMINO_ACIDS[token[0] - 3],
                        fontsize=36,
                        horizontalalignment="center",
                    )
                    # Draw the bar graph of predictions underneath
                    top5_preds = figure.add_subplot(
                        4,
                        len(tokenised) - 1,
                        (i + len(tokenised), i + 1 + 3 * (len(tokenised) - 1)),
                    )
                    colors = [
                        "red" if aa == AMINO_ACIDS[token[0] - 3] else "C0"
                        for aa in reversed(preds_collection[i + 1][0])
                    ]
                    top5_preds.barh(
                        range(5), reversed(preds_collection[i + 1][1]), color=colors
                    )
                    top5_preds.set_yticks(range(5))
                    top5_preds.set_yticklabels(reversed(preds_collection[i + 1][0]))
                    top5_preds.set_xlim(0, 1)
                figure.tight_layout()

                exemplar_figures[f"exemplar_{row_idx}.png"] = figure

            return mlm_summary, exemplar_figures

        if re.search("CDR(Cls)?BERT", model_class_name):
            # Quantify MLM performance
            total_acc = 0
            total_cdr1_acc = 0
            total_cdr2_acc = 0
            total_cdr3_acc = 0
            divisor = 0

            for x, y in tqdm(dataloader):
                num_samples = len(x)

                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model.model.mlm(x)

                total_acc += mlm_acc(logits, y) * num_samples
                total_cdr1_acc += mlm_acc(logits, y, x[:, :, 3] == 1) * num_samples
                total_cdr2_acc += mlm_acc(logits, y, x[:, :, 3] == 2) * num_samples
                total_cdr3_acc += mlm_acc(logits, y, x[:, :, 3] == 3) * num_samples

                divisor += num_samples

            mlm_summary = {
                "acc": total_acc / divisor,
                "cdr1_acc": total_cdr1_acc / divisor,
                "cdr2_acc": total_cdr2_acc / divisor,
                "cdr3_acc": total_cdr3_acc / divisor,
            }

            # Generate exemplar plots
            exemplars = self.bg_data.sample(5, random_state=420, ignore_index=True)
            exemplar_figures = dict()
            color_scheme = {
                "cdr1_symbol": "red",
                "cdr2_symbol": "blue",
                "cdr3_symbol": "green",
            }

            for row_idx, exemplar in exemplars.iterrows():
                # Tokenise
                tokenised = self.model._tokeniser.tokenise(exemplar).to(self.device)

                # Mask each residue and note mlm predictions
                preds_collection = dict()
                for residue_idx in range(1, len(tokenised)):
                    masked = tokenised.detach().clone()
                    masked[residue_idx, 0] = 0
                    logits = self.model.model.mlm(masked.unsqueeze(0))[0, residue_idx]
                    scores = softmax(logits, dim=0)
                    pred_scores, preds_indices = topk(scores, 5, dim=0)
                    preds = [AMINO_ACIDS[idx] for idx in preds_indices]
                    preds_collection[residue_idx] = (preds, pred_scores.detach().cpu())

                # Visualise as a figure
                figure = plt.figure(figsize=(len(tokenised) - 1, 5))
                # For each token in sequence
                for i, token in enumerate(tokenised[1:]):
                    # Write out the correct token
                    symbol = figure.add_subplot(4, len(tokenised) - 1, i + 1)
                    symbol.axis("off")
                    symbol.text(
                        0.5,
                        0,
                        AMINO_ACIDS[token[0] - 3],
                        color=color_scheme[f"cdr{token[3]}_symbol"],
                        fontsize=36,
                        horizontalalignment="center",
                    )
                    # Draw the bar graph of predictions underneath
                    top5_preds = figure.add_subplot(
                        4,
                        len(tokenised) - 1,
                        (i + len(tokenised), i + 1 + 3 * (len(tokenised) - 1)),
                    )
                    colors = [
                        "red" if aa == AMINO_ACIDS[token[0] - 3] else "C0"
                        for aa in reversed(preds_collection[i + 1][0])
                    ]
                    top5_preds.barh(
                        range(5), reversed(preds_collection[i + 1][1]), color=colors
                    )
                    top5_preds.set_yticks(range(5))
                    top5_preds.set_yticklabels(reversed(preds_collection[i + 1][0]))
                    top5_preds.set_xlim(0, 1)
                figure.tight_layout()

                exemplar_figures[f"exemplar_{row_idx}.png"] = figure

            return mlm_summary, exemplar_figures

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

    def benchmark_epitopes(self, ds_name: str, ds_df: DataFrame) -> tuple:
        embs = self.get_embs(ds_name, ds_df)
        epitopes = torch.tensor(ds_df["Epitope"].astype("category").cat.codes).to(
            self.device
        )

        knn_scores = BenchmarkingPipeline.knn(embs, epitopes)
        avg_precision, precisions, recalls = BenchmarkingPipeline.precision_recall(
            embs, epitopes
        )

        pr_curve, ax = plt.subplots()
        ax.step(recalls, precisions)
        ax.set_title(f"{ds_name} pr curve ({self.model.name})")
        ax.set_ylabel("Precision")
        ax.set_xlabel("Recall")

        performance_summary = {"knn_scores": knn_scores, "avg_precision": avg_precision}

        return performance_summary, precisions, recalls, pr_curve

    def setup(self, model_save_dir: Path) -> None:
        self.model = ModelLoader(model_save_dir)
        self.device = self.model._device

        self.benchmark_dir = PROJECT_DIR / "benchmarks_beta" / self.model.name
        self.cache_dir = self.benchmark_dir / ".cache"

        if not self.benchmark_dir.is_dir():
            self.benchmark_dir.mkdir()
        if not self.cache_dir.is_dir():
            self.cache_dir.mkdir()

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
