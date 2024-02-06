import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from sklearn import metrics
from sklearn.svm import SVC
from src.model_analyser.analysis import Analysis
from src.model_analyser.analysis_result import AnalysisResult
from src.model_analyser.tcr_edit_distance_records.tcr_edit import JunctionEdit
from src.model_analyser.tcr_edit_distance_records.tcr_edit_distance_record_collection import TcrEditDistanceRecordCollection
from src.model_analyser.tcr_edit_distance_records.tcr_edit_distance_record_collection_analyser import TcrEditDistanceRecordCollectionAnalyser
from src.model_analyser.tcr_edit_distance_records import tcr_edit_generator
from tqdm import tqdm
from typing import Iterable, Tuple


class OneVsBackground(Analysis):
    def run(self) -> AnalysisResult:
        results_dict = dict()

        bg_sample = self._background_data.sample(n=11_000, random_state=420)
        self.training_bg_sample = bg_sample[:1_000]
        self.test_bg_sample = bg_sample[1_000:]

        self.training_data_grouped_by_epitope = self._labelled_data["benchmarking_training"].groupby("Epitope")
        self.test_data_grouped_by_epitope = self._labelled_data["benchmarking_testing"].groupby("Epitope")

        results_dict["nn_classification"] = self._benchmark_nn_classification()
        results_dict["svc_classification"] = self._benchmark_svc_classification()

        return AnalysisResult("one_vs_background", results=results_dict)
    
    def _benchmark_nn_classification(self) -> dict:
        results_dict = dict()

        for epitope in tqdm(self.training_data_grouped_by_epitope.groups):
            test_positives = self.test_data_grouped_by_epitope.get_group(epitope)
            test_data = pd.concat([test_positives, self.test_bg_sample])

            ground_truth = test_data["Epitope"] == epitope
            predictive_distance = self._get_dists_to_nn(epitope, test_data)
            similarity_scores = np.exp(-predictive_distance/50)

            auc = metrics.roc_auc_score(ground_truth, similarity_scores)
            auc_01 = metrics.roc_auc_score(ground_truth, similarity_scores, max_fpr=0.1)

            results_dict[epitope] = {
                "reference_size": len(self.training_data_grouped_by_epitope.get_group(epitope)),
                "auc": auc,
                "auc0.1": auc_01,
            }
        
        return results_dict
    
    def _get_dists_to_nn(self, epitope: str, test_data: DataFrame) -> ndarray:
        epitope_reference = self.training_data_grouped_by_epitope.get_group(epitope)
        cdist_matrix = self._model_computation_cacher.calc_cdist_matrix(test_data, epitope_reference)
        return cdist_matrix.min(axis=1)

    def _benchmark_svc_classification(self) -> dict:
        results_dict = dict()

        characteristic_length = self._get_characteristic_distance()
        def rbf_kernel(cdist: ndarray, characteristic_length: float) -> ndarray:
            return np.exp(- (cdist / characteristic_length) ** 2)

        for epitope in tqdm(self.training_data_grouped_by_epitope.groups):
            training_positives = self.training_data_grouped_by_epitope.get_group(epitope)
            training_data = pd.concat([training_positives, self.training_bg_sample])
            training_cdist_matrix = self._model_computation_cacher.calc_cdist_matrix(training_data, training_data)
            training_kernel_matrix = rbf_kernel(training_cdist_matrix, characteristic_length)

            test_positives = self.test_data_grouped_by_epitope.get_group(epitope)
            test_data = pd.concat([test_positives, self.test_bg_sample])
            test_cdist_matrix = self._model_computation_cacher.calc_cdist_matrix(test_data, training_data)
            test_kernel_matrix = rbf_kernel(test_cdist_matrix, characteristic_length)

            training_target = (training_data["Epitope"] == epitope).to_numpy()
            
            svc = SVC(kernel="precomputed")
            svc.fit(training_kernel_matrix, training_target)

            test_target = test_data["Epitope"] == epitope
            test_preds = svc.decision_function(test_kernel_matrix)
            
            auc = metrics.roc_auc_score(test_target, test_preds)
            auc_01 = metrics.roc_auc_score(test_target, test_preds, max_fpr=0.1)

            results_dict[epitope] = {
                "reference_size": len(self.training_data_grouped_by_epitope.get_group(epitope)),
                "auc": auc,
                "auc0.1": auc_01,
            }
        
        return results_dict
    
    def _get_characteristic_distance(self) -> float:
        ed_record_collection = self._model_computation_cacher.get_tcr_edit_record_collection()
        ed_record_collection = self._catalogue_distances_of_junction_subs(ed_record_collection)
        self._model_computation_cacher.save_tcr_edit_record_collection(ed_record_collection)

        analyser = TcrEditDistanceRecordCollectionAnalyser(ed_record_collection)
        return analyser.get_average_distance_over_central_edits()

    def _catalogue_distances_of_junction_subs(self, ed_record_collection: TcrEditDistanceRecordCollection) -> TcrEditDistanceRecordCollection:
        num_tcrs_processed = 0

        while not ed_record_collection.has_sufficient_central_junction_edit_coverage():
            tcr = self._background_data.sample(n=1)
            edits_and_resulting_distances = self._get_all_central_subs_and_resulting_distances(tcr)

            for edit, distance in edits_and_resulting_distances:
                ed_record_collection.update_edit_record(edit, distance)

            num_tcrs_processed += 1

            if (num_tcrs_processed % 10) == 0:
                print(f"{num_tcrs_processed} TCRs processed...")
                ed_record_collection.print_current_estimation_coverage()

        return ed_record_collection
    
    def _get_all_central_subs_and_resulting_distances(
        self, tcr: DataFrame
    ) -> Iterable[Tuple[JunctionEdit, float]]:
        original_tcr = tcr
        junction_variants = tcr_edit_generator.get_junction_variants(tcr)
        central_variants = junction_variants[
            junction_variants["edit"].map(lambda x: x.is_central)
        ]

        distances_between_original_and_edited_tcrs = self._model.calc_cdist_matrix(
            original_tcr, central_variants
        ).squeeze()

        return zip(central_variants["edit"], distances_between_original_and_edited_tcrs)