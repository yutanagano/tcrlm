from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.spatial import distance
from sklearn import metrics
from src.model_analyser.analysis import Analysis
from src.model_analyser.analysis_result import AnalysisResult


class CospecificityAnalysis(Analysis):
    BG_SAMPLE_SIZE = 10_000

    def run(self, testing: bool = False) -> AnalysisResult:
        if testing:
            self.BG_SAMPLE_SIZE = 2

        results = dict()
        figures = dict()

        reference_data = self._labelled_data["benchmarking_training"]
        testing_data = self._labelled_data["benchmarking_testing"]

        cross_reference_distances = self._model_computation_cacher.calc_cdist_matrix(testing_data, reference_data)
        cross_reference_ground_truth = self._get_cospecificity_ground_truth(testing_data, reference_data)
        cross_reference_auc_stats = self._get_auc_stats(cross_reference_distances, cross_reference_ground_truth, testing_data["Epitope"])
        cross_reference_pr_stats = self._get_pr_stats(cross_reference_distances, cross_reference_ground_truth)

        results["cross_reference_auc"] = cross_reference_auc_stats
        results["cross_reference_pr"] = cross_reference_pr_stats
        figures["cross_reference_auc"] = self._plot_auc(cross_reference_auc_stats, "cross-reference")
        figures["cross_reference_pr"] = self._plot_pr(cross_reference_pr_stats, "cross-reference")

        within_testing_distances = self._model_computation_cacher.calc_cdist_matrix(testing_data, testing_data)
        within_testing_ground_truth = self._get_cospecificity_ground_truth(testing_data, testing_data)
        within_testing_auc_stats = self._get_auc_stats(within_testing_distances, within_testing_ground_truth, testing_data["Epitope"])
        within_testing_pr_stats = self._get_pr_stats(within_testing_distances, within_testing_ground_truth)

        results["within_testing_auc"] = within_testing_auc_stats
        results["within_testing_pr"] = within_testing_pr_stats
        figures["within_testing_auc"] = self._plot_auc(within_testing_auc_stats, "within-testing")
        figures["within_testing_pr"] = self._plot_pr(within_testing_pr_stats, "within-testing")

        return AnalysisResult("cospecificity", results=results, figures=figures)
    
    def _get_cospecificity_ground_truth(self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame) -> ndarray:
        anchor_epitope_catcodes = self._get_epitope_cat_codes(anchor_tcrs)
        comparison_epitope_catcodes = self._get_epitope_cat_codes(comparison_tcrs)
        return anchor_epitope_catcodes[:,np.newaxis] == comparison_epitope_catcodes[np.newaxis,:]

    def _get_epitope_cat_codes(self, dataset: DataFrame) -> ndarray:
        return dataset.Epitope.astype("category").cat.codes.to_numpy()

    def _get_auc_stats(self, distances: ndarray, ground_truth: ndarray, epitopes: Series) -> dict:
        results_dict = dict()

        similarity_scores = self._get_similarity_scores_from_distances(distances)

        for epitope in epitopes.unique():
            mask = (epitopes == epitope).to_numpy()
            similarity_scores_for_epitope = similarity_scores[mask, :].flatten()
            ground_truth_for_epitope = ground_truth[mask, :].flatten()

            fpr, tpr, _ = metrics.roc_curve(ground_truth_for_epitope, similarity_scores_for_epitope, drop_intermediate=True)
            auc_roc = metrics.roc_auc_score(ground_truth_for_epitope, similarity_scores_for_epitope)

            results_dict[epitope] = {
                "auc": auc_roc,
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist()
            }
        
        return results_dict
    
    def _get_pr_stats(self, distances: ndarray, ground_truth: ndarray) -> dict:
        distances = distances.flatten()
        ground_truth = ground_truth.flatten()
        similarity_scores = self._get_similarity_scores_from_distances(distances)
        precisions, recalls, _ = metrics.precision_recall_curve(ground_truth, similarity_scores, drop_intermediate=True)
        avg_precision = metrics.average_precision_score(ground_truth, similarity_scores)
        return {
            "avg_precision": avg_precision,
            "precisions": precisions.tolist(),
            "recalls": recalls.tolist()
        }

    def _get_similarity_scores_from_distances(self, distances: ndarray) -> ndarray:
        return np.exp(-distances / 50)
    
    def _plot_auc(self, auc_stats: dict, name: str) -> Figure:
        fig, ax = plt.subplots(figsize=(20,10))

        for epitope, stats in auc_stats.items():
            ax.step(stats["fpr"], stats["tpr"], label=f"{epitope} ({stats['auc']})")
            
        ax.set_title(f"ROC ({name})")
        ax.set_xlabel("fpr")
        ax.set_ylabel("tpr")
        ax.legend()

        fig.tight_layout()

        return fig
    
    def _plot_pr(self, pr_stats: dict, name: str) -> Figure:
        fig, ax = plt.subplots()

        ax.step(pr_stats["recalls"], pr_stats["precisions"])
        ax.set_title(f"PR ({name}): AP = {pr_stats['avg_precision']}")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")

        fig.tight_layout()

        return fig

    # def _evaluate_pr_curve(self, dataset: DataFrame) -> dict:
    #     dataset_expanded = self._expand_dataset_for_repeated_clones(dataset)

    #     pdist_vector = self._model_computation_cacher.calc_pdist_vector(
    #         dataset_expanded
    #     )
    #     epitope_cat_codes = self._get_epitope_cat_codes(dataset_expanded)

    #     similarity_scores = self._get_similarity_scores_from_distances(pdist_vector)
    #     positive_pair_mask = self._get_positive_pair_mask_from_epitope_cat_codes(
    #         epitope_cat_codes
    #     )

    #     precisions, recalls, thresholds = self._get_precision_recall_curve(
    #         positive_pair_mask, similarity_scores
    #     )

    #     if len(precisions) > 10_000:
    #         precisions = self._intelligently_subsample_list(precisions)
    #         recalls = self._intelligently_subsample_list(recalls)
    #         thresholds = self._intelligently_subsample_list(thresholds)

    #     background_discovery_rates = self._get_background_discovery_rates(
    #         thresholds, dataset
    #     )
    #     avg_precision = metrics.average_precision_score(
    #         positive_pair_mask, similarity_scores
    #     )

    #     return {
    #         "avg_precision": avg_precision,
    #         "thresholds": thresholds,
    #         "precisions": precisions,
    #         "recalls": recalls,
    #         "background_discovery_rates": background_discovery_rates,
    #     }

    # def _get_background_discovery_rates(
    #     self, thresholds: list, reference_tcrs: DataFrame
    # ) -> list:
    #     cdist_matrix = self._model_computation_cacher.calc_cdist_matrix(
    #         self._background_data.sample(n=self.BG_SAMPLE_SIZE, random_state=420),
    #         reference_tcrs,
    #     )
    #     similarity_scores = self._get_similarity_scores_from_distances(cdist_matrix)
    #     thresholds_except_infinitessimal = thresholds[1:]

    #     discovery_rates = [0]

    #     for threshold in thresholds_except_infinitessimal:
    #         last_discovery_rate = discovery_rates[-1]
    #         if last_discovery_rate == 1:
    #             discovery_rate = 1

    #         within_threshold = similarity_scores >= threshold
    #         discovery_rate = within_threshold.mean()

    #         discovery_rates.append(discovery_rate)

    #     return discovery_rates

    # def _plot_background_discovery_rate(
    #     self, pr_stats: dict, dataset_name: str
    # ) -> Figure:
    #     fig, ax = plt.subplots()

    #     ax.step(pr_stats["recalls"], pr_stats["background_discovery_rates"])
    #     ax.set_title(f"Background discovery rate vs Recall ({dataset_name})")
    #     ax.set_ylabel("Background discovery rate")
    #     ax.set_xlabel("Recall")
    #     ax.set_xscale("log")
    #     ax.set_yscale("log")
    #     fig.tight_layout()

    #     return fig