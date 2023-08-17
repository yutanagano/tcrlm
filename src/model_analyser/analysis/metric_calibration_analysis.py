import numpy as np
from numpy import ndarray
from pandas import DataFrame
import random
from tqdm import tqdm

from src.model_analyser.analysis import Analysis
from src.model_analyser.analysis_result import AnalysisResult


NUM_BG_DISTANCES_TO_SAMPLE = 1_000_000
BG_DISTANCE_SAMPLE_FILENAME = "bg_distances.txt"


class MetricCalibrationAnalysis(Analysis):
    def run(self) -> AnalysisResult:
        results_dict = dict()

        bg_dist_histogram = self._compute_background_distance_histogram()
        results_dict["bg_dist_hist"] = bg_dist_histogram

        for dataset_name, dataset in self._labelled_data.items():
            ep_matched_dist_histogram = self._compute_epitope_matched_distance_histogram_from_dataset(dataset)
            results_dict[f"ep_matched_dist_hist_{dataset_name}"] = ep_matched_dist_histogram

        return AnalysisResult("metric_calibration", results_dict=results_dict)

    def _compute_background_distance_histogram(self) -> ndarray:
        sample_of_distances = self._get_background_distance_sample()
        return self._generate_histogram_from_sample(sample_of_distances)
    
    def _get_background_distance_sample(self) -> ndarray:
        self._sample_background_distances_and_save_to_cache()
        with self._model_computation_cacher.get_readable_buffer(BG_DISTANCE_SAMPLE_FILENAME) as f:
            background_distance_sample = np.loadtxt(f)
        return background_distance_sample

    def _sample_background_distances_and_save_to_cache(self) -> None:
        num_distances_already_sampled = self._get_num_background_distances_already_sampled()
        num_distances_left_to_sample = NUM_BG_DISTANCES_TO_SAMPLE - num_distances_already_sampled
        if num_distances_left_to_sample > 0:
            self._sample_n_bg_distances_and_save_to_cache(n=num_distances_left_to_sample)
    
    def _get_num_background_distances_already_sampled(self) -> int:
        with self._model_computation_cacher.get_readable_buffer(BG_DISTANCE_SAMPLE_FILENAME) as f:
            num_lines_in_sample_file = sum(1 for line in f)
        return num_lines_in_sample_file
    
    def _sample_n_bg_distances_and_save_to_cache(self, n: int) -> None:
        bg_data_with_repeated_clones = self._expand_dataset_for_repeated_clones(self._background_data)

        with self._model_computation_cacher.get_appendable_buffer(BG_DISTANCE_SAMPLE_FILENAME) as f:
            for _ in tqdm(range(n)):
                anchor_index, comparison_index = random.sample(range(len(bg_data_with_repeated_clones)), k=2)
                relevant_tcrs = bg_data_with_repeated_clones.iloc[[anchor_index, comparison_index]]
                distance = self._model.calc_pdist_vector(relevant_tcrs).item()
                f.write(f"{distance}\n")
    
    def _compute_epitope_matched_distance_histogram_from_dataset(self, dataset: DataFrame) -> ndarray:
        epitope_matched_df_expanded = self._expand_dataset_for_repeated_clones(dataset)
        epitopes_in_dataset = epitope_matched_df_expanded.Epitope.unique().tolist()

        pdists = []
        for epitope in epitopes_in_dataset:
            tcrs_in_epitope_group = epitope_matched_df_expanded[epitope_matched_df_expanded.Epitope == epitope]
            ep_group_pdist_vector = self._model_computation_cacher.calc_pdist_vector(tcrs_in_epitope_group)
            pdists.append(ep_group_pdist_vector)

        collective_distance_sample = np.concatenate(pdists)

        return self._generate_histogram_from_sample(collective_distance_sample)

    def _expand_dataset_for_repeated_clones(self, dataset: DataFrame) -> DataFrame:
        index_expanding_repeated_clones = dataset.index.repeat(dataset.clone_count)
        return dataset.loc[index_expanding_repeated_clones]
    
    def _generate_histogram_from_sample(self, sample_of_distances: ndarray) -> ndarray:
        distance_bins = self._model.distance_bins
        histogram, _ = np.histogram(sample_of_distances, bins=distance_bins, density=True)
        return histogram