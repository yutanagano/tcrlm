from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series
from scipy.spatial import distance
import statistics
from sklearn import metrics
from typing import Dict

from src.model_analyser.analysis import Analysis
from src.model_analyser.analysis_result import AnalysisResult
from src.model.tcr_metric import BetaCdr3Levenshtein, BetaCdrLevenshtein


MINIMUM_NUM_PAIRS = 100


class AucByLevenshteinGroups(Analysis):
    _results: dict
    _figures: Dict[str, Figure]

    def run(self) -> AnalysisResult:
        if self._current_model_is_irrelevant_for_this_analysis():
            return AnalysisResult("na")

        self._setup()

        for dataset_name, dataset in self._labelled_data.items():
            cospecificity_labels = self._get_cospecificity_labels(dataset)
            beta_cdr3_levenshtein_pdist = self._get_beta_cdr3_levenshtein_pdist(dataset)
            model_pdist = self._get_model_pdist(dataset)
            dist_table = DataFrame(
                {
                    "cospecificity_label": cospecificity_labels,
                    "levenshtein": beta_cdr3_levenshtein_pdist,
                    "model": model_pdist,
                }
            )

            self._analyse_model_performance(dataset_name, dist_table)

        return AnalysisResult(
            "auc_by_levenshtein_groups", results=self._results, figures=self._figures
        )

    def _current_model_is_irrelevant_for_this_analysis(self) -> bool:
        return type(self._model) in (BetaCdr3Levenshtein, BetaCdrLevenshtein)

    def _setup(self) -> None:
        self._results = dict()
        self._figures = dict()
        self._beta_cdr3_levenshtein_model = BetaCdr3Levenshtein()

    def _get_beta_cdr3_levenshtein_pdist(self, dataset: DataFrame) -> ndarray:
        return self._beta_cdr3_levenshtein_model.calc_pdist_vector(dataset)

    def _get_cospecificity_labels(self, dataset: DataFrame) -> ndarray:
        epitopes_as_ndarray = dataset.Epitope.to_numpy()
        cospecificity_table = (
            epitopes_as_ndarray[:, np.newaxis] == epitopes_as_ndarray[np.newaxis, :]
        )
        return distance.squareform(cospecificity_table, checks=False)

    def _get_model_pdist(self, dataset: DataFrame) -> ndarray:
        return self._model_computation_cacher.calc_pdist_vector(dataset)

    def _analyse_model_performance(
        self, dataset_name: str, dist_table: DataFrame
    ) -> None:
        performance_dict = dict()

        max_levenshtein_distance = dist_table.levenshtein.max()
        for dist in range(1, max_levenshtein_distance + 1):
            if not self._enough_examples_at_levenshtein_distance(dist_table, dist):
                continue
            performance_dict[dist] = self._analyse_model_performance_at_given_dist(
                dist_table, dist
            )

        self._results[dataset_name] = performance_dict
        self._figures[f"roc_auc_{dataset_name}"] = self._generate_roc_auc_plot(
            dataset_name, performance_dict
        )

    def _enough_examples_at_levenshtein_distance(
        self, dist_table: DataFrame, dist: int
    ) -> bool:
        filtered_dist_table = dist_table[dist_table.levenshtein == dist]
        if filtered_dist_table.empty:
            return False

        cospecificity_group_sizes = filtered_dist_table.groupby(
            "cospecificity_label"
        ).size()
        return cospecificity_group_sizes.map(lambda x: x > MINIMUM_NUM_PAIRS).all()

    def _analyse_model_performance_at_given_dist(
        self, dist_table: DataFrame, dist: int
    ) -> dict:
        roc_aucs = []

        for i in range(10):
            dist_table_filtered = self._prepare_balanced_dist_table_for_distance(
                dist_table, dist
            )
            similarity_scores = self._get_similarity_scores_from_distances(
                dist_table_filtered.model
            )

            roc_auc = metrics.roc_auc_score(
                y_true=dist_table_filtered.cospecificity_label,
                y_score=similarity_scores,
            )

            roc_aucs.append(roc_auc)

        return roc_aucs

    def _prepare_balanced_dist_table_for_distance(
        self, dist_table: DataFrame, dist: int
    ) -> DataFrame:
        filtered_dist_table = dist_table[dist_table.levenshtein == dist]
        number_of_pairs_to_sample_from_each_group = (
            filtered_dist_table.groupby("cospecificity_label").size().min()
        )

        positive_pairs = filtered_dist_table[
            filtered_dist_table.cospecificity_label == True
        ]
        negative_pairs = filtered_dist_table[
            filtered_dist_table.cospecificity_label == False
        ]

        positive_pairs_subsampled = positive_pairs.sample(
            n=number_of_pairs_to_sample_from_each_group
        )
        negative_pairs_subsampled = negative_pairs.sample(
            n=number_of_pairs_to_sample_from_each_group
        )

        return pd.concat(
            [positive_pairs_subsampled, negative_pairs_subsampled]
        ).reset_index(drop=True)

    def _get_similarity_scores_from_distances(self, distances: Series) -> ndarray:
        return np.exp(-distances / 50)

    def _generate_roc_auc_plot(
        self, dataset_name: str, performance_dict: dict
    ) -> Figure:
        dists = performance_dict.keys()
        roc_auc_lists = [performance_dict[dist] for dist in dists]
        mean_roc_aucs = np.array(
            [statistics.mean(roc_aucs) for roc_aucs in roc_auc_lists]
        )
        standard_deviations = np.array(
            [statistics.stdev(roc_aucs) for roc_aucs in roc_auc_lists]
        )

        fig, ax = plt.subplots()

        ax.plot(dists, mean_roc_aucs)
        ax.fill_between(
            dists,
            mean_roc_aucs + standard_deviations,
            mean_roc_aucs - standard_deviations,
            alpha=0.2,
        )

        ax.set_ylim(0, 1)

        ax.set_xlabel("Levenshtein distance")
        ax.set_ylabel("ROC-AUC")
        ax.set_title(f"ROC-AUC by Levenshtein distance groups ({dataset_name})")

        return fig
