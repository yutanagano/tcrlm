from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from scipy.spatial import distance
from sklearn import metrics

from src.model_analyser.analysis import Analysis
from src.model_analyser.analysis_result import AnalysisResult


class PrecisionRecallAnalysis(Analysis):
    def run(self) -> AnalysisResult:
        results_dict = dict()
        figures = dict()

        for dataset_name, dataset in self._labelled_data.items():
            pr_stats = self._evaluate_pr_curve(dataset)
            pr_figure = self._plot_pr_curve(pr_stats, dataset_name)

            results_dict[f"avg_precision_{dataset_name}"] = pr_stats["avg_precision"]
            figures[f"{dataset_name}_pr_curve"] = pr_figure

        return AnalysisResult(
            "precision_recall", results=results_dict, figures=figures
        )

    def _evaluate_pr_curve(self, dataset: DataFrame) -> dict:
        pdist_vector = self._model_computation_cacher.calc_pdist_vector(dataset)
        epitope_cat_codes = self._get_epitope_cat_codes(dataset)

        similarity_scores = self._get_similarity_scores_from_distances(pdist_vector)
        positive_pair_mask = self._get_positive_pair_mask_from_epitope_cat_codes(
            epitope_cat_codes
        )

        precisions, recalls, _ = metrics.precision_recall_curve(
            positive_pair_mask, similarity_scores
        )
        avg_precision = metrics.average_precision_score(
            positive_pair_mask, similarity_scores
        )

        return {
            "avg_precision": avg_precision,
            "precisions": precisions.tolist(),
            "recalls": recalls.tolist(),
        }

    def _get_epitope_cat_codes(self, dataset: DataFrame) -> ndarray:
        return dataset.Epitope.astype("category").cat.codes.to_numpy()

    def _get_similarity_scores_from_distances(self, distances: ndarray) -> ndarray:
        return np.exp(-distances / 50)

    def _get_positive_pair_mask_from_epitope_cat_codes(
        self, epitope_cat_codes: ndarray
    ) -> ndarray:
        where_comparisons_are_between_same_epitope_group = (
            epitope_cat_codes[:, None] == epitope_cat_codes[None, :]
        )
        return distance.squareform(
            where_comparisons_are_between_same_epitope_group, checks=False
        )

    def _plot_pr_curve(self, pr_stats: dict, dataset_name: str) -> Figure:
        fig, ax = plt.subplots()

        ax.step(pr_stats["recalls"], pr_stats["precisions"])
        ax.set_title(f"Precision-Recall Curve ({dataset_name})")
        ax.set_ylabel("Precision")
        ax.set_xlabel("Recall")
        fig.tight_layout()

        return fig
