import numpy as np
from numpy import ndarray
from pandas import DataFrame
from scipy import stats
from scipy.spatial import distance

from src.model_analyser.analysis import Analysis
from src.model_analyser.analysis_result import AnalysisResult


class KnnAnalysis(Analysis):
    def run(self) -> AnalysisResult:
        results_dict = dict()

        for dataset_name, dataset in self._labelled_data.items():
            knn_stats = self._evaluate_knn_accuracy(dataset)
            results_dict[dataset_name] = knn_stats

        return AnalysisResult("knn", results=results_dict)

    def _evaluate_knn_accuracy(self, dataset: DataFrame) -> dict:
        pdist_vector = self._model_computation_cacher.calc_pdist_vector(dataset)
        pdist_matrix = distance.squareform(pdist_vector)
        pdist_matrix = pdist_matrix.astype(float)
        epitope_cat_codes = self._get_epitope_cat_codes(dataset)

        np.fill_diagonal(pdist_matrix, np.inf)

        knn_stats = dict()

        for k in (5, 10, 50, 100):
            scores = []
            size = len(dataset)

            for tcr_index in range(size):
                correct_epitope_label = epitope_cat_codes[tcr_index]
                distances_to_other_tcrs = pdist_matrix[tcr_index]
                indices_of_k_nearest_tcrs = np.argsort(distances_to_other_tcrs)[:k]

                epitopes_of_k_nearest_tcrs = epitope_cat_codes[
                    indices_of_k_nearest_tcrs
                ]
                pred, _ = stats.mode(
                    epitopes_of_k_nearest_tcrs, keepdims=True
                )  # Predict epitope
                scores.append(
                    correct_epitope_label.item() == pred.item()
                )  # Record score

            score = np.array(scores, dtype=np.float32).mean().item()
            knn_stats[k] = score

        return knn_stats

    def _get_epitope_cat_codes(self, dataset: DataFrame) -> ndarray:
        return dataset.Epitope.astype("category").cat.codes.to_numpy()
