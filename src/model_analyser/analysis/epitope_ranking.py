import numpy as np
from numpy import ndarray
from pandas import DataFrame
from sklearn import metrics
from src.model_analyser.analysis import Analysis
from src.model_analyser.analysis_result import AnalysisResult
import statistics
from tqdm import tqdm


class EpitopeRanking(Analysis):
    def run(self) -> AnalysisResult:
        results_dict = dict()

        reference_data = self._labelled_data["benchmarking_training"]
        test_data = self._labelled_data["benchmarking_testing"]
        test_data_epitope_catcodes = test_data["Epitope"].astype("category").cat.codes.to_numpy()[:,np.newaxis]

        epitopes = reference_data["Epitope"].unique()
        dists_to_nn = []
        for epitope in tqdm(epitopes):
            predictive_distance = self._get_dists_to_nn(test_data, reference_data, epitope)
            dists_to_nn.append(predictive_distance)
        dists_to_nn = np.stack(dists_to_nn)
        epitopes_ranked = np.argsort(dists_to_nn, axis=0).T
        epitope_rankings = np.nonzero(epitopes_ranked == test_data_epitope_catcodes)[1]

        avg_ranks = []
        for epitope in epitopes:
            mask = (test_data["Epitope"] == epitope).to_numpy()
            rankings = epitope_rankings[mask]
            avg_rank = rankings.mean()
            avg_ranks.append(avg_rank)

        results_dict["avg_rank"] = statistics.mean(avg_ranks)

        return AnalysisResult("epitope_ranking", results=results_dict)
    
    def _get_dists_to_nn(self, test_data: DataFrame, reference_data: DataFrame, epitope: str) -> ndarray:
        epitope_reference = reference_data[reference_data["Epitope"] == epitope]
        cdist_matrix = self._model_computation_cacher.calc_cdist_matrix(test_data, epitope_reference)
        return cdist_matrix.min(axis=1)