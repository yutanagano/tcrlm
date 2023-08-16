from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray
from tqdm import tqdm

from src.model_analyser.analysis import Analysis
from src.model_analyser.analysis_result import AnalysisResult


class PgenAnalysis(Analysis):
    def run(self) -> AnalysisResult:
        bg_tcrs_avg_dist_to_100_nearest_neighbours = self._get_bg_tcrs_avg_dist_to_100_nearest_neighbours()
        pgen_vs_local_density_figure = self._generate_pgen_vs_local_density_figure(bg_tcrs_avg_dist_to_100_nearest_neighbours)
        figures = {
            "pgen_vs_local_density": pgen_vs_local_density_figure
        }

        return AnalysisResult("pgen_vs_local_density", figures=figures)
    
    def _get_bg_tcrs_avg_dist_to_100_nearest_neighbours(self) -> ndarray:
        return self._model_computation_cacher.get_cached_or_compute_array(
            "bg_avg_dist_to_100_nn.npy",
            self._avg_dist_computation
        )

    def _avg_dist_computation(self) -> ndarray:
        BATCH_SIZE = 1000
        KTH = min(100, len(self._background_data)-1)

        range_over_indices_of_first_row_of_batches = range(0, len(self._background_data), BATCH_SIZE)

        avg_dists = []

        for i in tqdm(range_over_indices_of_first_row_of_batches):
            dists_batch = self._model_computation_cacher.calc_cdist_matrix(
                self._background_data.iloc[i:i+BATCH_SIZE],
                self._background_data
            ).squeeze()

            for dists in dists_batch:
                dists_to_closest_100 = np.partition(dists, kth=KTH)[:KTH]
                avg_dist = dists_to_closest_100.mean()
                avg_dists.append(avg_dist)

        return np.array(avg_dists, dtype=np.float32)
    
    def _generate_pgen_vs_local_density_figure(self, bg_tcrs_avg_dist_to_100_nearest_neighbours: ndarray) -> Figure:
        log10_pgens = np.log10(self._background_pgen["beta_pgen"])
        bins = range(-15, -7)
        bin_positions = range(len(bins) + 1)
        avg_dists_by_pgen = self._bin_avg_dists_by_pgen(bg_tcrs_avg_dist_to_100_nearest_neighbours, log10_pgens, bins)

        fig, ax = plt.subplots()
        ax.violinplot(avg_dists_by_pgen, positions=bin_positions)

        bin_labels = ["$<10^{" + str(bins[0]) + "}$"]
        bin_labels += [
            "$10^{" + str(bins[i]) + "}-10^{" + str(bins[i+1]) + "}$" for i in range(len(bins)-1)
        ]
        bin_labels += ["$>10^{" + str(bins[-1]) + "}$"]

        ax.set_xticks(bin_positions)
        ax.set_xticklabels(bin_labels, rotation=45, ha="right")

        fig.tight_layout()

        return fig
    
    def _bin_avg_dists_by_pgen(self, avg_dists, pgens, bins) -> list:
        binned_data = [[] for _ in range(len(bins) + 1)]

        inds = np.digitize(pgens, bins)
        for value, ind in zip(avg_dists, inds):
            binned_data[ind].append(value)

        return binned_data