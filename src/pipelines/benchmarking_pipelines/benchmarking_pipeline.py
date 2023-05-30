"""
Base benchmarking pipelines.
"""


from ...datahandling.datasets import TCRDataset
from ...datahandling.dataloaders import MLMDataLoader
from ...metrics import mlm_acc
from ...model_loader import ModelLoader
from ...resources import AMINO_ACIDS
from ..class_method_metaclass import ClassMethodMeta
from argparse import ArgumentParser
import json
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from pathlib import Path
import re
from scipy.spatial.distance import squareform
from scipy.stats import mode
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_curve
import torch
from torch import device, Tensor, topk
from torch.nn import Linear, Module, SoftMarginLoss
from torch.nn.functional import softmax
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable, Dict, Optional, Tuple


class BenchmarkingPipeline(metaclass=ClassMethodMeta):
    BACKGROUND_DATA_PATH = "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/tanno/test.csv"
    BACKGROUND_PGEN_PATH = "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/tanno/test_pgens.npy"
    LABELLED_DATA_PATHS = {
        "vdjdb": "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/vdjdb/evaluation_beta.csv",
        "dash": "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/dash/evaluation.csv",
        "mira": "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/mira/valid.csv",
    }

    model: ModelLoader
    torch_device: device
    background_data: DataFrame
    labelled_data: Dict[str, DataFrame]
    summary_dict: Dict
    figures: Dict[str, Figure]
    save_dir: Path

    def run_from_clargs(cls) -> None:
        parser = ArgumentParser(description="Benchmarking pipeline.")
        parser.add_argument("model_save_dir", help="Path to model save directory.")
        args = parser.parse_args()

        cls.main(Path(args.model_save_dir))

    def main(cls, model_save_dir: Path) -> None:
        cls.setup(model_save_dir)
        cls.benchmark_mlm_performance()
        cls.explore_embedding_space()
        cls.evaluate_pgen_vs_representation_space_density()
        for ds_name, ds_df in cls.labelled_data.items():
            cls.becnhmark_on_labelled_data(ds_name, ds_df)
        cls.save()
        print("Done!")

    def setup(cls, model_save_dir: Path) -> None:
        print("Setting up...")
        cls.instantiate_model(model_save_dir)
        cls.load_data()
        cls.instantiate_results_dictionaries()
        cls.prepare_save_directory()
        cls.set_plotting_theme()

    def instantiate_model(cls, model_save_dir: Path) -> None:
        cls.model = ModelLoader(model_save_dir)
        cls.torch_device = cls.model._device

    def load_data(cls) -> None:
        cls.background_data = cls.load_csv(cls.BACKGROUND_DATA_PATH)
        cls.background_data["Epitope"] = "BG"
        cls.background_pgen = np.load(cls.BACKGROUND_PGEN_PATH)
        cls.labelled_data = {
            name: cls.load_csv(path) for name, path in cls.LABELLED_DATA_PATHS.items()
        }

    @staticmethod
    def load_csv(path: Path) -> DataFrame:
        return pd.read_csv(path)

    def instantiate_results_dictionaries(cls) -> None:
        cls.summary_dict = {"model_name": cls.model.name}
        cls.figures = dict()

    def prepare_save_directory(cls) -> None:
        benchmark_dir = Path.cwd() / "benchmarks_beta"
        if not benchmark_dir.is_dir():
            benchmark_dir.mkdir()

        cls.save_dir = benchmark_dir / cls.model.name
        if not cls.save_dir.is_dir():
            cls.save_dir.mkdir()

        cache_dir = cls.save_dir / ".cache"
        if not cache_dir.is_dir():
            cache_dir.mkdir()

    @staticmethod
    def set_plotting_theme() -> None:
        sns.set_theme()

    def benchmark_mlm_performance(cls) -> None:
        print("Benchmarking MLM performance...")

        model_class_name = cls.model.model.__class__.__name__
        dataloader = MLMDataLoader(
            dataset=TCRDataset(
                data=cls.background_data, tokeniser=cls.model._tokeniser
            ),
            batch_size=512,
            shuffle=False,
            p_mask=0.01,
            p_mask_random=0,
            p_mask_keep=0,
        )
        exemplar_tcrs = cls.background_data.sample(
            5, random_state=420, ignore_index=True
        )

        if re.search("BCDR3(Cls)?BERT", model_class_name):
            cls.benchmark_cdr3_mlm_performance(dataloader, exemplar_tcrs)
            return

        if re.search("BCDR(Cls)?BERT", model_class_name):
            cls.benchmark_cdr123_mlm_performance(dataloader, exemplar_tcrs)
            return

    def benchmark_cdr3_mlm_performance(
        cls, dataloader: MLMDataLoader, exemplar_tcrs: DataFrame
    ) -> None:
        mlm_performance = cls.get_cached(
            "mlm.json",
            lambda: cls.quantify_cdr3_mlm_performance(dataloader)
        )
        mlm_exemplar_figures = cls.generate_cdr3_mlm_exemplar_plots(exemplar_tcrs)

        cls.summary_dict["mlm_summary"] = mlm_performance
        cls.figures = {**cls.figures, **mlm_exemplar_figures}

    def benchmark_cdr123_mlm_performance(
        cls, dataloader: MLMDataLoader, exemplar_tcrs: DataFrame
    ) -> None:
        mlm_performance = cls.get_cached(
            "mlm.json",
            lambda: cls.quantify_cdr123_mlm_performance(dataloader)
        )
        mlm_exemplar_figures = cls.generate_cdr123_mlm_exemplar_plots(exemplar_tcrs)

        cls.summary_dict["mlm_summary"] = mlm_performance
        cls.figures = {**cls.figures, **mlm_exemplar_figures}

    def quantify_cdr3_mlm_performance(
        cls, dataloader: MLMDataLoader
    ) -> Dict[str, float]:
        total_acc = 0
        divisor = 0

        for x, y in tqdm(dataloader):
            num_samples = len(x)

            x = x.to(cls.torch_device)
            y = y.to(cls.torch_device)

            logits = cls.model.model.mlm(x)

            total_acc += mlm_acc(logits, y) * num_samples
            divisor += num_samples

        return {"acc": total_acc / divisor}

    def quantify_cdr123_mlm_performance(
        cls, dataloader: MLMDataLoader
    ) -> Dict[str, float]:
        total_acc = 0
        total_cdr1_acc = 0
        total_cdr2_acc = 0
        total_cdr3_acc = 0
        divisor = 0

        for x, y in tqdm(dataloader):
            num_samples = len(x)

            x = x.to(cls.torch_device)
            y = y.to(cls.torch_device)

            logits = cls.model.model.mlm(x)

            total_acc += mlm_acc(logits, y) * num_samples
            total_cdr1_acc += mlm_acc(logits, y, x[:, :, 3] == 1) * num_samples
            total_cdr2_acc += mlm_acc(logits, y, x[:, :, 3] == 2) * num_samples
            total_cdr3_acc += mlm_acc(logits, y, x[:, :, 3] == 3) * num_samples
            divisor += num_samples

        return {
            "acc": total_acc / divisor,
            "cdr1_acc": total_cdr1_acc / divisor,
            "cdr2_acc": total_cdr2_acc / divisor,
            "cdr3_acc": total_cdr3_acc / divisor,
        }

    def generate_cdr3_mlm_exemplar_plots(
        cls, exemplar_tcrs: DataFrame
    ) -> Dict[str, Figure]:
        mlm_exemplar_figures = dict()

        for row_idx, exemplar in exemplar_tcrs.iterrows():
            tokenised = cls.model._tokeniser.tokenise(exemplar).to(cls.torch_device)
            model_preds_collection = cls.get_model_preds(tokenised)

            cdr3_length = len(tokenised) - 1  # Subtract one for <cls>

            fig = plt.figure(figsize=(cdr3_length, 5))

            for token_pos, token in enumerate(tokenised[1:], 1):
                correct_token_id = token[0] - 3
                model_preds = model_preds_collection[token_pos]
                cls.draw_mlm_exemplar_column(
                    fig, token_pos, cdr3_length, correct_token_id, model_preds
                )

            fig.tight_layout()

            mlm_exemplar_figures[f"exemplar_{row_idx}"] = fig

        return mlm_exemplar_figures

    def generate_cdr123_mlm_exemplar_plots(
        cls, exemplar_tcrs: DataFrame
    ) -> Dict[str, Figure]:
        mlm_exemplar_figures = dict()
        color_scheme = {
            "cdr1_symbol": "red",
            "cdr2_symbol": "blue",
            "cdr3_symbol": "green",
        }

        for row_idx, exemplar in exemplar_tcrs.iterrows():
            tokenised = cls.model._tokeniser.tokenise(exemplar).to(cls.torch_device)
            model_preds_collection = cls.get_model_preds(tokenised)

            tcr_length = len(tokenised) - 1  # Subtract one for <cls>

            fig = plt.figure(figsize=(tcr_length, 5))

            for token_pos, token in enumerate(tokenised[1:], 1):
                correct_token_id = token[0] - 3
                model_preds = model_preds_collection[token_pos]
                text_color = color_scheme[f"cdr{token[3]}_symbol"]
                cls.draw_mlm_exemplar_column(
                    fig,
                    token_pos,
                    tcr_length,
                    correct_token_id,
                    model_preds,
                    text_color,
                )

            fig.tight_layout()

            mlm_exemplar_figures[f"exemplar_{row_idx}"] = fig

        return mlm_exemplar_figures

    def get_model_preds(cls, tokenised: Tensor) -> Dict[int, tuple]:
        model_preds = dict()

        for residue_idx in range(1, len(tokenised)):
            masked = tokenised.detach().clone()
            masked[residue_idx, 0] = 0
            logits = cls.model.model.mlm(masked.unsqueeze(0))[0, residue_idx]
            scores = softmax(logits, dim=0)
            pred_scores, pred_indices = topk(scores, 5, dim=0)
            pred_scores = pred_scores.detach().cpu().flip(0)
            pred_indices = pred_indices.detach().cpu().flip(0)
            model_preds[residue_idx] = (pred_indices, pred_scores)

        return model_preds

    def draw_mlm_exemplar_column(
        cls,
        fig: Figure,
        token_pos: int,
        cdr3_length: int,
        correct_token_id: int,
        model_preds: tuple,
        text_color: str = "black",
    ) -> None:
        # Write out the correct token
        symbol = fig.add_subplot(4, cdr3_length, token_pos)
        symbol.axis("off")
        symbol.text(
            0.5,
            0,
            AMINO_ACIDS[correct_token_id],
            color=text_color,
            fontsize=36,
            horizontalalignment="center",
        )

        # Draw the bar graph of predictions underneath
        top5_preds = fig.add_subplot(
            4,
            cdr3_length,
            (token_pos + cdr3_length, token_pos + 3 * cdr3_length),
        )
        colors = ["red" if aa == correct_token_id else "C0" for aa in model_preds[0]]
        top5_preds.barh(range(5), model_preds[1], color=colors)
        top5_preds.set_yticks(range(5))
        top5_preds.set_yticklabels([AMINO_ACIDS[i] for i in model_preds[0]])
        top5_preds.set_xlim(0, 1)

    def explore_embedding_space(cls) -> None:
        print("Exploring embedding space using background data...")

        background_embs = cls.get_embeddings("tanno", cls.background_data)

        pca = PCA()
        pca.fit(background_embs.cpu())

        pca_summary_figure = cls.generate_pca_summary_figure(pca)
        pca_projection_figure = cls.generate_pca_2d_projection(pca, background_embs)
        pca_projection_bv = cls.generate_pca_2d_projection(pca, background_embs, cls.background_data["TRBV"])

        cls.figures["pca_summary"] = pca_summary_figure
        cls.figures["pca_projection"] = pca_projection_figure
        cls.figures["pca_projection_bv"] = pca_projection_bv

    def get_embeddings(cls, ds_name: str, ds: DataFrame) -> Tensor:
        return cls.get_cached(
            f"{ds_name}_embs.pt",
            lambda: torch.tensor(cls.model.embed(ds))
        )
    
    def get_cached(cls, filename: str, compute_result_fn: Callable) -> any:
        save_path = cls.save_dir / ".cache" / filename
        suffix = save_path.suffix

        if suffix == ".pt":
            return cls.get_cached_torch_tensor(save_path, compute_result_fn)
        if suffix == ".npy":
            return cls.get_cached_numpy_ndarray(save_path, compute_result_fn)
        if suffix == ".json":
            return cls.get_cached_json(save_path, compute_result_fn)
    
    def get_cached_torch_tensor(cls, save_path: Path, compute_result_fn: Callable) -> Tensor:
        if save_path.is_file():
            return torch.load(save_path).to(cls.torch_device)
        
        computed_result = compute_result_fn()
        torch.save(computed_result.cpu(), save_path)

        return computed_result.to(cls.torch_device)
    
    def get_cached_numpy_ndarray(cls, save_path: Path, compute_result_fn: Callable) -> ndarray:
        if save_path.is_file():
            return np.load(save_path)
        
        computed_result = compute_result_fn()
        np.save(save_path, computed_result)

        return computed_result
    
    def get_cached_json(cls, save_path: Path, compute_result_fn: Callable) -> dict:
        if save_path.is_file():
            with open(save_path, "r") as f:
                return json.load(f)
            
        computed_result = compute_result_fn()
        with open(save_path, "w") as f:
            json.dump(computed_result, f)
        
        return computed_result

    def generate_pca_summary_figure(cls, pca: PCA) -> Figure:
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        with sns.axes_style("dark"):
            fig, ax1 = plt.subplots()

            ax1.set_title(f"PCA Summary ({cls.model.name})")
            ax1.set_ylabel("Variance per PCA (bars)")

            ax2 = ax1.twinx()
            ax2.set_ylabel("Cumulative Variance (line)")
            ax2.set_ylim(0, 1.05)

            ax1.bar(range(pca.n_components_), pca.explained_variance_ratio_)
            ax2.plot(cumulative_variance, c="C1")

            fig.tight_layout()

        return fig

    @staticmethod
    def generate_pca_2d_projection(pca: PCA, embs: Tensor, categories: Optional[ndarray] = None) -> Figure:
        projection = pca.transform(embs.cpu())[:10000, :2]
        if not categories is None:
            categories = categories[:10000]

        with sns.axes_style("dark"):
            joint_grid = sns.jointplot(x=projection[:, 0], y=projection[:, 1], hue=categories)
            joint_grid.set_axis_labels(xlabel="PCA 1", ylabel="PCA 2")
            if not categories is None:
                joint_grid.ax_joint.legend_.remove()

        return joint_grid
    
    def evaluate_pgen_vs_representation_space_density(cls) -> None:
        print("Estimating representation space densities for different pGens...")

        avg_dist_to_100nn = cls.get_cached(
            "tanno_avg_dist_to_100nn.npy",
            cls.get_avg_dist_to_100nn_over_background
        )
        
        bins = range(-15, -7)
        log10_pgens = np.log10(cls.background_pgen)
        avg_dists_by_pgen = cls.bin_data(avg_dist_to_100nn, log10_pgens, bins)

        pgen_dist_figure = cls.generate_pgen_dist_figure(avg_dists_by_pgen, bins)

        cls.figures["acg_dist_vs_pgen"] = pgen_dist_figure

    def get_avg_dist_to_100nn_over_background(cls) -> ndarray:
        background_embs = cls.get_embeddings("tanno", cls.background_data)

        avg_dists = []

        for emb in tqdm(background_embs):
            dists = (background_embs - emb.unsqueeze(0)).norm(dim=1)
            dists_to_closest_100 = np.partition(dists.detach().cpu(), kth=100)[:100]
            avg_dist = dists_to_closest_100.mean()
            avg_dists.append(avg_dist)

        return np.array(avg_dists, dtype=np.float32)

    @staticmethod
    def bin_data(values, binnables, bins) -> list:
        binned_data = [[] for _ in range(len(bins) + 1)]

        inds = np.digitize(binnables, bins)
        for value, ind in zip(values, inds):
            binned_data[ind].append(value)

        return binned_data

    def generate_pgen_dist_figure(cls, dists_by_pgen, bins) -> Figure:
        fig, ax = plt.subplots()
        ax.violinplot(dists_by_pgen, positions=range(len(bins) + 1))

        bin_labels = ["$<10^{" + str(bins[0]) + "}$"]
        bin_labels += [
            "$10^{" + str(bins[i]) + "}-10^{" + str(bins[i+1]) + "}$" for i in range(len(bins)-1)
        ]
        bin_labels += ["$>10^{" + str(bins[-1]) + "}$"]

        ax.set_xticks(range(len(bins) + 1))
        ax.set_xticklabels(bin_labels, rotation=45, ha="right")

        fig.tight_layout()

        return fig

    def becnhmark_on_labelled_data(cls, ds_name: str, ds_df: DataFrame) -> None:
        print(f"Benchmarking on {ds_name}...")

        pdist_matrix = cls.get_pdist_matrix(ds_name, ds_df)
        epitope_cat_codes = cls.get_column_cat_codes(ds_df, "Epitope")

        knn_scores = cls.evaluate_knn_performance(pdist_matrix, epitope_cat_codes)
        avg_precision, precisions, recalls = cls.evaluate_precision_recall_curve(
            pdist_matrix, epitope_cat_codes
        )

        pr_figure = cls.generate_precision_recall_figure(precisions, recalls, ds_name)

        svm_dict = cls.evaluate_svm_performance(ds_df)

        cls.summary_dict[ds_name] = {
            "knn_scores": knn_scores,
            "avg_precision": avg_precision,
            "svm_performance": svm_dict
        }
        cls.figures[f"{ds_name}_pr_curve"] = pr_figure

    def get_pdist_matrix(cls, ds_name: str, ds_df: DataFrame) -> tuple:
        embs = cls.get_embeddings(ds_name, ds_df)
        pdist_array = torch.pdist(embs, p=2).detach().cpu()
        pdist_matrix = squareform(pdist_array)

        return pdist_matrix
    
    @staticmethod
    def get_column_cat_codes(df: DataFrame, column: str) -> ndarray:
        return df[column].astype("category").cat.codes.to_numpy()

    @staticmethod
    def evaluate_knn_performance(
        cdist_matrix: ndarray, epitope_cat_codes: ndarray
    ) -> Dict[str, float]:
        cdist_cp = cdist_matrix.copy()
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
    def evaluate_precision_recall_curve(
        cdist_matrix: ndarray, epitope_cat_codes: ndarray
    ) -> Tuple[float, ndarray]:
        pdist = squareform(cdist_matrix)
        probs = np.exp(-pdist / 50)
        positive_pair = (epitope_cat_codes[:, None] == epitope_cat_codes[None, :]) & (
            np.eye(len(epitope_cat_codes)) != 1
        )
        positive_pair = squareform(positive_pair)

        precisions, recalls, _ = precision_recall_curve(positive_pair, probs)
        avg_precision = average_precision_score(positive_pair, probs)

        return avg_precision, precisions, recalls

    def generate_precision_recall_figure(
        cls, precisions: ndarray, recalls: ndarray, ds_name: str
    ) -> Figure:
        fig, ax = plt.subplots()
        ax.step(recalls, precisions)
        ax.set_title(f"{ds_name} pr curve ({cls.model.name})")
        ax.set_ylabel("Precision")
        ax.set_xlabel("Recall")
        fig.tight_layout()

        return fig
    
    def evaluate_svm_performance(cls, ds_df: DataFrame) -> dict:
        epitopes = ds_df["Epitope"].unique()

        results_dict = dict()

        for epitope in epitopes:
            ep_train = ds_df[ds_df["Epitope"] == epitope][:50]
            ep_valid = ds_df[ds_df["Epitope"] == epitope][50:]
            bg_train = cls.background_data.sample(n=50, random_state=420)
            bg_valid = cls.background_data.sample(n=50, random_state=421)

            train = pd.concat([ep_train, bg_train], ignore_index=True)
            train = train[["TRBV", "CDR3B", "TRBJ", "Epitope"]]

            valid = pd.concat([ep_valid, bg_valid], ignore_index=True)
            valid = valid[["TRBV", "CDR3B", "TRBJ", "Epitope"]]

            train_labels = train.Epitope.map({epitope: 1, "BG": -1})
            valid_labels = valid.Epitope.map({epitope: 1, "BG": -1})

            train_tcrs = cls.model.embed(train)
            valid_tcrs = cls.model.embed(valid)

            model = SVM(cls.model.d_model)
            loss_fn = SoftMarginLoss()
            optimiser = Adam(params=model.parameters())

            train_dl = DataLoader(dataset=list(zip(train_tcrs, train_labels)), batch_size=100, shuffle=True)
            valid_dl = DataLoader(dataset=list(zip(valid_tcrs, valid_labels)), batch_size=100, shuffle=True)

            valid_acc_tracker = []
            train_acc_tracker = []

            for epoch in range(1000):
                valid_accuracies = []

                for tcrs, labels in valid_dl:
                    preds = model(tcrs)
                    labels = labels.unsqueeze(-1)

                    accuracy = ((preds * labels) > 0).to(float).mean()
                    valid_accuracies.append(accuracy.item())

                valid_acc_tracker.append(torch.tensor(valid_accuracies).mean().item())

                train_accuracies = []

                for tcrs, labels in train_dl:
                    preds = model(tcrs)
                    labels = labels.unsqueeze(-1)

                    optimiser.zero_grad()
                    loss = loss_fn(preds, labels)
                    loss.backward()
                    optimiser.step()

                    accuracy = ((preds * labels) > 0).to(float).mean()
                    train_accuracies.append(accuracy.item())

                train_acc_tracker.append(torch.tensor(train_accuracies).mean().item())

            results_dict[epitope] = {
                "max_acc": max(train_acc_tracker),
                "max_valid_acc": max(valid_acc_tracker)
            }
        
        return results_dict

    def save(cls) -> None:
        print("Saving...")

        with open(cls.save_dir / "summary.json", "w") as f:
            json.dump(cls.summary_dict, f, indent=4)

        for filename, plot in cls.figures.items():
            plot.savefig(cls.save_dir / filename)


class SVM(Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = Linear(d_model, 1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class BetaBenchmarkingPipeline(BenchmarkingPipeline):
    @staticmethod
    def load_csv(path: Path) -> DataFrame:
        df = pd.read_csv(path)
        df[["TRAV", "CDR3A", "TRAJ"]] = pd.NA
        return df
