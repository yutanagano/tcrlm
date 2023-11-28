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
    BG_SAMPLE_SIZE = 10_000

    def run(self, testing: bool = False) -> AnalysisResult:
        if testing:
            self.BG_SAMPLE_SIZE = 2

        results_dict = dict()
        figures = dict()

        for dataset_name, dataset in self._labelled_data.items():
            pr_stats = self._evaluate_pr_curve(dataset)
            pr_figure = self._plot_pr_curve(pr_stats, dataset_name)
            bg_discovery_rate_figure = self._plot_background_discovery_rate(
                pr_stats, dataset_name
            )

            results_dict[f"pr_stats_{dataset_name}"] = pr_stats
            figures[f"{dataset_name}_pr_curve"] = pr_figure
            figures[f"{dataset_name}_bg_discovery_rates"] = bg_discovery_rate_figure

        return AnalysisResult("precision_recall", results=results_dict, figures=figures)

    def _evaluate_pr_curve(self, dataset: DataFrame) -> dict:
        dataset_expanded = self._expand_dataset_for_repeated_clones(dataset)

        pdist_vector = self._model_computation_cacher.calc_pdist_vector(
            dataset_expanded
        )
        epitope_cat_codes = self._get_epitope_cat_codes(dataset_expanded)

        similarity_scores = self._get_similarity_scores_from_distances(pdist_vector)
        positive_pair_mask = self._get_positive_pair_mask_from_epitope_cat_codes(
            epitope_cat_codes
        )

        precisions, recalls, thresholds = self._get_precision_recall_curve(
            positive_pair_mask, similarity_scores
        )

        if len(precisions) > 10_000:
            precisions = self._intelligently_subsample_list(precisions)
            recalls = self._intelligently_subsample_list(recalls)
            thresholds = self._intelligently_subsample_list(thresholds)

        background_discovery_rates = self._get_background_discovery_rates(
            thresholds, dataset
        )
        avg_precision = metrics.average_precision_score(
            positive_pair_mask, similarity_scores
        )

        return {
            "avg_precision": avg_precision,
            "thresholds": thresholds,
            "precisions": precisions,
            "recalls": recalls,
            "background_discovery_rates": background_discovery_rates,
        }

    def _expand_dataset_for_repeated_clones(self, dataset: DataFrame) -> DataFrame:
        index_expanding_repeated_clones = dataset.index.repeat(dataset.clone_count)
        return dataset.loc[index_expanding_repeated_clones]

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

    def _get_precision_recall_curve(
        self, positive_pair_mask: ndarray, similarity_scores: ndarray
    ) -> (list, list, list):
        precisions, recalls, thresholds = metrics.precision_recall_curve(
            positive_pair_mask, similarity_scores
        )
        precisions = list(reversed(precisions))
        recalls = list(reversed(recalls))
        infinitessimal_threshold = 1.0
        thresholds = [infinitessimal_threshold] + list(
            reversed(thresholds.astype(float))
        )

        return (precisions, recalls, thresholds)

    def _intelligently_subsample_list(self, l: list) -> list:
        length = len(l)

        first_chunk_size = int(length * 0.001)
        second_chunk_size = int(length * 0.01)
        third_chunk_size = int(length * 0.1)

        first_chunk_boundary = first_chunk_size
        second_chunk_boundary = first_chunk_boundary + second_chunk_size
        third_chunk_boundary = second_chunk_boundary + third_chunk_size

        first_chunk = l[:first_chunk_boundary]
        second_chunk = l[first_chunk_boundary:second_chunk_boundary]
        third_chunk = l[second_chunk_boundary:third_chunk_boundary]
        remaining_points = l[third_chunk_boundary:]

        first_chunk = self._subsample_to_about_n_elements(first_chunk, 500)
        second_chunk = self._subsample_to_about_n_elements(second_chunk, 500)
        third_chunk = self._subsample_to_about_n_elements(third_chunk, 500)
        remaining_points = self._subsample_to_about_n_elements(remaining_points, 500)

        return first_chunk + second_chunk + third_chunk + remaining_points

    def _subsample_to_about_n_elements(self, l: list, n: int) -> list:
        list_length = len(l)
        skip_size = int(list_length / n)

        if skip_size == 0:
            return l

        remainder = list_length % skip_size

        subsampled = l[::skip_size]

        if remainder != 1:
            subsampled.append(l[-1])

        return subsampled

    def _get_background_discovery_rates(
        self, thresholds: list, reference_tcrs: DataFrame
    ) -> list:
        cdist_matrix = self._model_computation_cacher.calc_cdist_matrix(
            self._background_data.sample(n=self.BG_SAMPLE_SIZE, random_state=420),
            reference_tcrs,
        )
        similarity_scores = self._get_similarity_scores_from_distances(cdist_matrix)
        thresholds_except_infinitessimal = thresholds[1:]

        discovery_rates = [0]

        for threshold in thresholds_except_infinitessimal:
            last_discovery_rate = discovery_rates[-1]
            if last_discovery_rate == 1:
                discovery_rate = 1

            within_threshold = similarity_scores >= threshold
            discovery_rate = within_threshold.mean()

            discovery_rates.append(discovery_rate)

        return discovery_rates

    def _plot_pr_curve(self, pr_stats: dict, dataset_name: str) -> Figure:
        fig, ax = plt.subplots()

        ax.step(pr_stats["recalls"], pr_stats["precisions"])
        ax.set_title(f"Precision-Recall Curve ({dataset_name})")
        ax.set_ylabel("Precision")
        ax.set_xlabel("Recall")
        ax.set_xscale("log")
        ax.set_yscale("log")
        fig.tight_layout()

        return fig

    def _plot_background_discovery_rate(
        self, pr_stats: dict, dataset_name: str
    ) -> Figure:
        fig, ax = plt.subplots()

        ax.step(pr_stats["recalls"], pr_stats["background_discovery_rates"])
        ax.set_title(f"Background discovery rate vs Recall ({dataset_name})")
        ax.set_ylabel("Background discovery rate")
        ax.set_xlabel("Recall")
        ax.set_xscale("log")
        ax.set_yscale("log")
        fig.tight_layout()

        return fig
