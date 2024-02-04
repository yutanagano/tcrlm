import numpy as np
from numpy import ndarray
from sklearn import metrics
from sklearn.svm import SVC
from src.model.tcr_representation_model import TcrRepresentationModel
from src.model_analyser.analysis import Analysis
from src.model_analyser.analysis_result import AnalysisResult
from tqdm import tqdm


class OneVsRest(Analysis):
    def run(self) -> AnalysisResult:
        results_dict = dict()

        self.training_data = self._labelled_data["benchmarking_training"]
        self.test_data = self._labelled_data["benchmarking_testing"]

        results_dict["nn_classification"] = self._benchmark_nn_classification()
        results_dict["svc_classification"] = self._benchmark_svc_classification()

        return AnalysisResult("one_vs_rest", results=results_dict)
    
    def _benchmark_nn_classification(self) -> dict:
        results_dict = dict()

        epitopes = self.training_data["Epitope"].unique()
        for epitope in tqdm(epitopes):
            ground_truth = self.test_data["Epitope"] == epitope
            predictive_distance = self._get_dists_to_nn(epitope)
            similarity_scores = np.exp(-predictive_distance/50)

            auc = metrics.roc_auc_score(ground_truth, similarity_scores)
            auc_01 = metrics.roc_auc_score(ground_truth, similarity_scores, max_fpr=0.1)

            results_dict[epitope] = {
                "reference_size": int((self.training_data["Epitope"] == epitope).sum()),
                "auc": auc,
                "auc0.1": auc_01,
            }
        
        return results_dict
    
    def _get_dists_to_nn(self, epitope: str) -> ndarray:
        epitope_reference = self.training_data[self.training_data["Epitope"] == epitope]
        cdist_matrix = self._model_computation_cacher.calc_cdist_matrix(self.test_data, epitope_reference)
        return cdist_matrix.min(axis=1)

    def _benchmark_svc_classification(self) -> dict:
        results_dict = dict()

        training_cdist_matrix = self._model_computation_cacher.calc_cdist_matrix(self.training_data, self.training_data)
        test_cdist_matrix = self._model_computation_cacher.calc_cdist_matrix(self.test_data, self.training_data)
        cospecific_distances = self._get_cospecific_dists()
        characteristic_length = np.median(cospecific_distances)
        print(characteristic_length)

        def rbf_kernel(cdist: ndarray, characteristic_length: float) -> ndarray:
            return np.exp(- (cdist / characteristic_length) ** 2)
        
        training_kernel_matrix = rbf_kernel(training_cdist_matrix, characteristic_length)
        test_kernel_matrix = rbf_kernel(test_cdist_matrix, characteristic_length)

        epitopes = self.training_data["Epitope"].unique()
        for epitope in tqdm(epitopes):
            training_target = (self.training_data["Epitope"] == epitope).to_numpy()
            
            svc = SVC(kernel="precomputed")
            svc.fit(training_kernel_matrix, training_target)

            test_target = self.test_data["Epitope"] == epitope
            test_preds = svc.decision_function(test_kernel_matrix)
            
            auc = metrics.roc_auc_score(test_target, test_preds)
            auc_01 = metrics.roc_auc_score(test_target, test_preds, max_fpr=0.1)

            results_dict[epitope] = {
                "reference_size": int((self.training_data["Epitope"] == epitope).sum()),
                "auc": auc,
                "auc0.1": auc_01,
            }
        
        return results_dict
    
    def _get_cospecific_dists(self) -> ndarray:
        pdists = []
        training_data_groupby = self.training_data.groupby("Epitope")

        epitopes = self.training_data["Epitope"].unique()
        for epitope in epitopes:
            training_data_for_epitope = training_data_groupby.get_group(epitope)
            pdist = self._model_computation_cacher.calc_pdist_vector(training_data_for_epitope)
            pdists.append(pdist)
        
        return np.concatenate(pdists)