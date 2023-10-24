import math
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm
from typing import Dict, Iterable

from src.model_analyser.analysis import Analysis
from src.model_analyser.analysis_result import AnalysisResult


NUM_BG_DISTANCES_TO_SAMPLE = 100_000_000
BATCH_SIZE_WHEN_ESTIMATING_BG_DISTANCES = 1_000
BG_DISTANCE_SAMPLE_FILENAME = "bg_distances.txt"


class EmpiricalPdf:
    def __init__(self, densities: ndarray, bins: ndarray) -> None:
        self.densities = densities
        self.bins = bins

    def to_dict(self) -> Dict[str, list]:
        return {"densities": self.densities.tolist(), "bins": self.bins.tolist()}


class EmpiricalFunction:
    def __init__(self, x_coords: ndarray, y_coords: ndarray) -> None:
        self.x_coords = x_coords
        self.y_coords = y_coords

    def to_dict(self) -> Dict[str, list]:
        return {"x_coords": self.x_coords.tolist(), "y_coords": self.y_coords.tolist()}


class MetricCalibrationAnalysis(Analysis):
    def run(self) -> AnalysisResult:
        self._set_up_results_dicts()
        self._characterise_background_distance_distribution()

        for dataset_name, dataset in self._labelled_data.items():
            cospecific_dist_sample = (
                self._get_epitope_matched_distance_sample_from_dataset(dataset)
            )
            cospecific_dist_pdf = self._generate_distance_pdf_from_sample(
                cospecific_dist_sample
            )
            cospecific_dist_cdf = self._generate_distance_cdf_from_sample(
                cospecific_dist_sample
            )

            self._compare_cospecific_distance_distribution_to_background(
                cospecific_dist_pdf, cospecific_dist_cdf, dataset_name
            )
            self._run_deorphanisation_analysis(
                dataset, cospecific_dist_cdf, dataset_name
            )

        return AnalysisResult(
            "metric_calibration", results=self._results, figures=self._figures
        )

    def _set_up_results_dicts(self) -> None:
        self._results = dict()
        self._figures = dict()

    def _characterise_background_distance_distribution(self) -> None:
        bg_dist_sample = self._get_background_distance_sample()

        self._bg_dist_pdf = self._generate_distance_pdf_from_sample(bg_dist_sample)
        self._bg_dist_cdf = self._generate_distance_cdf_from_sample(bg_dist_sample)

        self._results["bg_dist_pdf"] = self._bg_dist_pdf.to_dict()
        self._results["bg_dist_cdf"] = self._bg_dist_cdf.to_dict()

    def _compare_cospecific_distance_distribution_to_background(
        self,
        cospecific_dist_pdf: EmpiricalPdf,
        cospecific_dist_cdf: EmpiricalFunction,
        dataset_name: str,
    ) -> None:
        pdf_plot = self._plot_pdf(self._bg_dist_pdf, cospecific_dist_pdf)
        cdf_plot = self._plot_cdf(self._bg_dist_cdf, cospecific_dist_cdf)
        pdf_ratio_plot = self._plot_pdf_ratio(self._bg_dist_pdf, cospecific_dist_pdf)
        cdf_ratio_plot = self._plot_cdf_ratio(self._bg_dist_cdf, cospecific_dist_cdf)

        enrichment_recall_plot = self._plot_enrichment_recall_curve(
            self._bg_dist_cdf, cospecific_dist_cdf
        )

        self._results[
            f"ep_matched_dist_pdf_{dataset_name}"
        ] = cospecific_dist_pdf.to_dict()
        self._results[
            f"ep_matched_dist_cdf_{dataset_name}"
        ] = cospecific_dist_cdf.to_dict()

        self._figures[f"pdf_{dataset_name}"] = pdf_plot
        self._figures[f"cdf_{dataset_name}"] = cdf_plot
        self._figures[f"pdf_ratio_{dataset_name}"] = pdf_ratio_plot
        self._figures[f"cdf_ratio_{dataset_name}"] = cdf_ratio_plot
        self._figures[f"enrichment_recall_plot_{dataset_name}"] = enrichment_recall_plot

    def _run_deorphanisation_analysis(
        self,
        dataset: DataFrame,
        cospecific_dist_cdf: EmpiricalFunction,
        dataset_name: str,
    ) -> None:
        pdist_matrix = self._get_pdist_matrix_with_diagonal_and_cross_epitope_distances_set_to_inifinite(
            dataset
        )
        deorphanisation_rate = self._get_deorphanisation_rate(pdist_matrix)

        deorphanisation_rate_plot = self._plot_deorphanisation_rate(
            deorphanisation_rate, cospecific_dist_cdf
        )

        self._results[
            f"deorphanisation_rate_{dataset_name}"
        ] = deorphanisation_rate.to_dict()
        self._figures[
            f"deorphanisation_rate_plot_{dataset_name}"
        ] = deorphanisation_rate_plot

    def _get_pdist_matrix_with_diagonal_and_cross_epitope_distances_set_to_inifinite(
        self, dataset: DataFrame
    ) -> ndarray:
        dataset_expanded_for_repeated_clones = self._expand_dataset_for_repeated_clones(
            dataset
        )
        pdist_matrix = self._model_computation_cacher.calc_cdist_matrix(
            dataset_expanded_for_repeated_clones, dataset_expanded_for_repeated_clones
        )
        pdist_matrix = pdist_matrix.astype(np.float32)

        diagonal_mask = np.eye(len(dataset_expanded_for_repeated_clones))
        cross_epitope_mask = self._get_cross_epitope_mask(
            dataset_expanded_for_repeated_clones
        )
        infinity_mask = np.logical_or(diagonal_mask, cross_epitope_mask)

        pdist_matrix[infinity_mask] = np.inf

        return pdist_matrix

    def _get_cross_epitope_mask(self, dataset: DataFrame) -> ndarray:
        epitope_array = dataset.Epitope.values
        cospecificity_mask = (
            epitope_array[:, np.newaxis] == epitope_array[np.newaxis, :]
        )
        return np.logical_not(cospecificity_mask)

    def _get_deorphanisation_rate(self, pdist_matrix: ndarray) -> EmpiricalFunction:
        distances = np.array(self._model.distance_bins)
        deorphanisation_rate = np.zeros_like(distances, dtype=np.float32)

        for index, distance in enumerate(distances):
            num_neighbours_per_tcr_at_given_distance = np.sum(
                pdist_matrix <= distance, axis=1
            )
            has_at_least_one_neighbour = num_neighbours_per_tcr_at_given_distance >= 1
            fraction_with_at_least_one_neighbour = np.mean(has_at_least_one_neighbour)
            deorphanisation_rate[index] = fraction_with_at_least_one_neighbour

        return EmpiricalFunction(x_coords=distances, y_coords=deorphanisation_rate)

    def _plot_deorphanisation_rate(
        self,
        deorphanisation_rate: EmpiricalFunction,
        cospecific_dist_cdf: EmpiricalFunction,
    ) -> Figure:
        relative_contour = cospecific_dist_cdf.y_coords / self._bg_dist_cdf.y_coords
        indices_to_make_relative_countour_monotonic = np.argsort(relative_contour)

        x_coords = relative_contour[indices_to_make_relative_countour_monotonic]
        y_coords = deorphanisation_rate.y_coords[
            indices_to_make_relative_countour_monotonic
        ]

        fig, ax = plt.subplots()

        ax.plot(x_coords, y_coords)

        ax.set_xlabel("Fold enrichment of co-specific pairs")
        ax.set_ylabel("Deorphanisation rate")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks(
            [1.0, 0.9, 0.5, 0.25, 0.1],
            labels=[
                "$10^0$",
                r"$9 \times 10^{-1}$",
                r"$5 \times 10^{-1}$",
                r"$2.5 \times 10^{-1}$",
                r"$1 \times 10^{-1}$",
            ],
        )
        fig.tight_layout()

        return fig

    def _get_background_distance_sample(self) -> ndarray:
        self._sample_background_distances_and_save_to_cache()
        with self._model_computation_cacher.get_readable_buffer(
            BG_DISTANCE_SAMPLE_FILENAME
        ) as f:
            background_distance_sample = np.loadtxt(f)
        return background_distance_sample

    def _sample_background_distances_and_save_to_cache(self) -> None:
        num_distances_already_sampled = (
            self._get_num_background_distances_already_sampled()
        )
        num_distances_left_to_sample = (
            NUM_BG_DISTANCES_TO_SAMPLE - num_distances_already_sampled
        )
        if num_distances_left_to_sample > 0:
            self._sample_n_bg_distances_and_save_to_cache(
                n=num_distances_left_to_sample
            )

    def _get_num_background_distances_already_sampled(self) -> int:
        with self._model_computation_cacher.get_readable_buffer(
            BG_DISTANCE_SAMPLE_FILENAME
        ) as f:
            num_lines_in_sample_file = sum(1 for line in f)
        return num_lines_in_sample_file

    def _sample_n_bg_distances_and_save_to_cache(self, n: int) -> None:
        bg_data_with_repeated_clones = self._expand_dataset_for_repeated_clones(
            self._background_data
        )
        num_distances_per_batch = math.comb(BATCH_SIZE_WHEN_ESTIMATING_BG_DISTANCES, 2)
        num_batches_to_sample = math.ceil(n / num_distances_per_batch)

        with self._model_computation_cacher.get_appendable_buffer(
            BG_DISTANCE_SAMPLE_FILENAME
        ) as f:
            for _ in tqdm(range(num_batches_to_sample)):
                relevant_tcrs = bg_data_with_repeated_clones.sample(
                    n=BATCH_SIZE_WHEN_ESTIMATING_BG_DISTANCES
                )
                distances = self._model.calc_pdist_vector(relevant_tcrs)
                f.writelines([f"{distance}\n" for distance in distances])

    def _get_epitope_matched_distance_sample_from_dataset(
        self, dataset: DataFrame
    ) -> ndarray:
        epitope_matched_df_expanded = self._expand_dataset_for_repeated_clones(dataset)
        epitopes_in_dataset = epitope_matched_df_expanded.Epitope.unique().tolist()

        pdists = []
        for epitope in epitopes_in_dataset:
            tcrs_in_epitope_group = epitope_matched_df_expanded[
                epitope_matched_df_expanded.Epitope == epitope
            ]
            ep_group_pdist_vector = self._model_computation_cacher.calc_pdist_vector(
                tcrs_in_epitope_group
            )
            pdists.append(ep_group_pdist_vector)

        collective_distance_sample = np.concatenate(pdists)

        return collective_distance_sample

    def _expand_dataset_for_repeated_clones(self, dataset: DataFrame) -> DataFrame:
        index_expanding_repeated_clones = dataset.index.repeat(dataset.clone_count)
        return dataset.loc[index_expanding_repeated_clones]

    def _generate_distance_pdf_from_sample(
        self, sample_of_distances: ndarray
    ) -> EmpiricalPdf:
        densities, bins = np.histogram(
            sample_of_distances, bins=self._model.distance_bins, density=True
        )
        return EmpiricalPdf(densities, bins)

    def _generate_distance_cdf_from_sample(
        self, sample_of_distances: ndarray
    ) -> EmpiricalFunction:
        sample_size = len(sample_of_distances)
        x_coords = np.array(self._model.distance_bins)

        cdf = np.zeros_like(self._model.distance_bins, dtype=float)
        for index, x_coord in enumerate(x_coords):
            num_dists_leq_x_coord = (sample_of_distances <= x_coord).sum()
            cdf[index] = num_dists_leq_x_coord / sample_size

        return EmpiricalFunction(x_coords=x_coords, y_coords=cdf)

    def _get_dxs_from_bins(self, bins: Iterable) -> ndarray:
        bins_as_array = np.array(bins)
        return bins_as_array[1:] - bins_as_array[:-1]

    def _plot_pdf(
        self, bg_dist_pdf: EmpiricalPdf, ep_matched_dist_pdf: EmpiricalPdf
    ) -> Figure:
        fig, ax = plt.subplots()

        ax.stairs(bg_dist_pdf.densities, bg_dist_pdf.bins, label="background")
        ax.stairs(
            ep_matched_dist_pdf.densities, ep_matched_dist_pdf.bins, label="co-specific"
        )

        ax.legend()
        ax.set_xlabel("Distance")
        ax.set_ylabel("Probability density")
        ax.set_yscale("log")
        fig.tight_layout()

        return fig

    def _plot_cdf(
        self, bg_dist_cdf: EmpiricalFunction, ep_matched_dist_cdf: EmpiricalFunction
    ) -> Figure:
        fig, ax = plt.subplots()

        ax.plot(bg_dist_cdf.x_coords, bg_dist_cdf.y_coords, label="background")
        ax.plot(
            ep_matched_dist_cdf.x_coords,
            ep_matched_dist_cdf.y_coords,
            label="co-specific",
        )

        ax.legend()
        ax.set_xlabel("Distance")
        ax.set_ylabel(r"$P(d \leq x)$")
        ax.set_yscale("log")
        fig.tight_layout()

        return fig

    def _plot_pdf_ratio(
        self, bg_dist_pdf: EmpiricalPdf, ep_matched_dist_pdf: EmpiricalPdf
    ) -> Figure:
        bins = ep_matched_dist_pdf.bins
        relative_countour = ep_matched_dist_pdf.densities / bg_dist_pdf.densities

        fig, ax = plt.subplots()

        ax.stairs(relative_countour, bins)

        ax.set_xlabel("Distance")
        ax.set_ylabel(r"$1/P(cospecific) \times P(cospecific|d)$")
        ax.set_yscale("log")
        fig.tight_layout()

        return fig

    def _plot_cdf_ratio(
        self, bg_dist_cdf: EmpiricalFunction, ep_matched_dist_cdf: EmpiricalFunction
    ) -> Figure:
        x_coords = ep_matched_dist_cdf.x_coords
        relative_countour = ep_matched_dist_cdf.y_coords / bg_dist_cdf.y_coords

        fig, ax = plt.subplots()

        ax.plot(x_coords, relative_countour)

        ax.set_xlabel("Distance threshold")
        ax.set_ylabel("Fold enrichment of co-specific pairs")
        ax.set_yscale("log")
        fig.tight_layout()

        return fig

    def _plot_enrichment_recall_curve(
        self, bg_dist_cdf: EmpiricalFunction, ep_matched_dist_cdf: EmpiricalFunction
    ) -> Figure:
        relative_contour = ep_matched_dist_cdf.y_coords / bg_dist_cdf.y_coords
        indices_to_make_relative_countour_monotonic = np.argsort(relative_contour)

        x_coords = relative_contour[indices_to_make_relative_countour_monotonic]
        y_coords = ep_matched_dist_cdf.y_coords[
            indices_to_make_relative_countour_monotonic
        ]

        fig, ax = plt.subplots()

        ax.plot(x_coords, y_coords)

        ax.set_xlabel("Fold enrichment of co-specific pairs")
        ax.set_ylabel("Fraction of co-specific pairs captured")
        ax.set_xscale("log")
        ax.set_yscale("log")
        fig.tight_layout()

        return fig
