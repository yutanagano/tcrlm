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
        results_dict = dict()
        figures_dict = dict()

        bg_dist_sample = self._get_background_distance_sample()
        bg_dist_pdf = self._generate_pdf_from_sample(bg_dist_sample)
        bg_dist_cdf = self._generate_cdf_from_sample(bg_dist_sample)
        results_dict["bg_dist_pdf"] = bg_dist_pdf.to_dict()
        results_dict["bg_dist_cdf"] = bg_dist_cdf.to_dict()

        for dataset_name, dataset in self._labelled_data.items():
            ep_matched_dist_sample = (
                self._get_epitope_matched_distance_sample_from_dataset(dataset)
            )
            ep_matched_dist_pdf = self._generate_pdf_from_sample(ep_matched_dist_sample)
            ep_matched_dist_cdf = self._generate_cdf_from_sample(ep_matched_dist_sample)

            pdf_plot = self._plot_pdf(bg_dist_pdf, ep_matched_dist_pdf)
            cdf_plot = self._plot_cdf(bg_dist_cdf, ep_matched_dist_cdf)
            pdf_ratio_plot = self._plot_pdf_ratio(bg_dist_pdf, ep_matched_dist_pdf)
            cdf_ratio_plot = self._plot_cdf_ratio(bg_dist_cdf, ep_matched_dist_cdf)

            enrichment_recall_plot = self._plot_enrichment_recall_curve(
                bg_dist_cdf, ep_matched_dist_cdf
            )

            results_dict[
                f"ep_matched_dist_pdf_{dataset_name}"
            ] = ep_matched_dist_pdf.to_dict()
            results_dict[
                f"ep_matched_dist_cdf_{dataset_name}"
            ] = ep_matched_dist_cdf.to_dict()

            figures_dict[f"pdf_{dataset_name}"] = pdf_plot
            figures_dict[f"cdf_{dataset_name}"] = cdf_plot
            figures_dict[f"pdf_ratio_{dataset_name}"] = pdf_ratio_plot
            figures_dict[f"cdf_ratio_{dataset_name}"] = cdf_ratio_plot
            figures_dict[
                f"enrichment_recall_plot_{dataset_name}"
            ] = enrichment_recall_plot

        return AnalysisResult(
            "metric_calibration", results_dict=results_dict, figures=figures_dict
        )

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

    def _generate_pdf_from_sample(self, sample_of_distances: ndarray) -> EmpiricalPdf:
        densities, bins = np.histogram(
            sample_of_distances, bins=self._model.distance_bins, density=True
        )
        return EmpiricalPdf(densities, bins)

    def _generate_cdf_from_sample(
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
        ax.set_ylabel("$PDF(d)$")
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
        ax.set_ylabel(r"$P(cospecific|d) \times P(cospecific)^{-1}$")
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

        ax.set_xlabel("Distance")
        ax.set_ylabel(r"$P(cospecific|d \leq x) \times P(cospecific)^{-1}$")
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

        ax.set_xlabel("Fold enrichment of co-specific pairs from background")
        ax.set_ylabel("Fraction of co-specific pairs captured")
        ax.set_xscale("log")
        ax.set_yscale("log")
        fig.tight_layout()

        return fig
