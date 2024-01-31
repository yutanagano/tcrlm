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
        if isinstance(self._model, TcrRepresentationModel):
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

    def _benchmark_svc_classification(self) -> dict:
        results_dict = dict()
        training_tcr_representations = self._model_computation_cacher.calc_vector_representations(self.training_data)
        test_tcr_representations = self._model_computation_cacher.calc_vector_representations(self.test_data)

        epitopes = self.training_data["Epitope"].unique()
        for epitope in tqdm(epitopes):
            training_target = self.training_data["Epitope"] == epitope
            svc = SVC()
            svc.fit(training_tcr_representations, training_target)

            test_target = self.test_data["Epitope"] == epitope
            test_preds = svc.decision_function(test_tcr_representations)
            
            auc = metrics.roc_auc_score(test_target, test_preds)
            auc_01 = metrics.roc_auc_score(test_target, test_preds, max_fpr=0.1)

            results_dict[epitope] = {
                "reference_size": int((self.training_data["Epitope"] == epitope).sum()),
                "auc": auc,
                "auc0.1": auc_01,
            }
        
        return results_dict
    
    def _representation_classifier_on_epitope(self, epitope: str) -> dict:
        # if torch.cuda.is_available():
        #     device = torch.device("cuda:0")
        # else:
        #     device = torch.device("cpu")
        
        # tcr_representations = torch.tensor(
        #     self._model_computation_cacher.calc_vector_representations(self.training_data),
        #     device=device
        # )

        # target = torch.tensor(
        #     self.training_data["Epitope"] == epitope,
        #     dtype=torch.long,
        #     device=device
        # )
        # fraction_positive = target.sum().item() / len(target)

        # simple_classifier = Linear(in_features=self._model.d_model, out_features=2).to(device=device)
        # criterion = CrossEntropyLoss(weight=torch.tensor([fraction_positive, 1-fraction_positive]))
        # optimiser = Adam(simple_classifier.parameters())
        
        # for i in range(5_000):
        #     optimiser.zero_grad()
        #     preds = simple_classifier(tcr_representations)
        #     loss = criterion(preds, target)
        #     loss.backward()
        #     optimiser.step()

        # test_tcr_representations = torch.tensor(
        #     self._model_computation_cacher.calc_vector_representations(self.test_data),
        #     device=device
        # )
        # test_target = self.test_data["Epitope"] == epitope
        # with torch.no_grad():
        #     test_preds = simple_classifier(test_tcr_representations)[:,1]
        pass

    
    def _get_dists_to_nn(self, epitope: str) -> ndarray:
        epitope_reference = self.training_data[self.training_data["Epitope"] == epitope]
        cdist_matrix = self._model_computation_cacher.calc_cdist_matrix(self.test_data, epitope_reference)
        return cdist_matrix.min(axis=1)