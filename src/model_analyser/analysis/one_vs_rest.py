import numpy as np
from numpy import ndarray
from pandas import DataFrame
from sklearn import metrics
from src.model_analyser.analysis import Analysis
from src.model_analyser.analysis_result import AnalysisResult
from tqdm import tqdm


class OneVsRest(Analysis):
    def run(self) -> AnalysisResult:
        results_dict = dict()

        reference_data = self._labelled_data["benchmarking_training"]
        test_data = self._labelled_data["benchmarking_testing"]

        epitopes = reference_data["Epitope"].unique()

        for epitope in tqdm(epitopes):
            ground_truth = test_data["Epitope"] == epitope
            predictive_distance = self._get_dists_to_nn(test_data, reference_data, epitope)
            similarity_scores = np.exp(-predictive_distance/50)

            tpr, fpr, thresholds = metrics.roc_curve(ground_truth, similarity_scores)
            auc = metrics.roc_auc_score(ground_truth, similarity_scores)
            auc_01 = metrics.roc_auc_score(ground_truth, similarity_scores, max_fpr=0.1)

            results_dict[epitope] = {
                "reference_size": int((reference_data["Epitope"] == epitope).sum()),
                "auc": auc,
                "auc0.1": auc_01,
                # "tpr": list(tpr),
                # "fpr": list(fpr)
            }

        return AnalysisResult("one_vs_rest", results=results_dict)
    
    def _get_dists_to_nn(self, test_data: DataFrame, reference_data: DataFrame, epitope: str) -> ndarray:
        epitope_reference = reference_data[reference_data["Epitope"] == epitope]
        cdist_matrix = self._model_computation_cacher.calc_cdist_matrix(test_data, epitope_reference)
        return cdist_matrix.min(axis=1)