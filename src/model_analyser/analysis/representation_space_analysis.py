from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray
from pandas import Series
from sklearn.decomposition import PCA
from src.model.tcr_representation_model import TcrRepresentationModel
from src.model_analyser.analysis import Analysis
from src.model_analyser.analysis_result import AnalysisResult


class RepresentationSpaceAnalysis(Analysis):
    def run(self) -> AnalysisResult:
        if not isinstance(self._model, TcrRepresentationModel):
            return AnalysisResult("na")

        self._set_up_result_dicts()
        self._characterise_background_tcr_distribution()

        for dataset_name, dataset in self._labelled_data.items():
            pass

        return AnalysisResult(
            "representation_space", results=self._results, figures=self._figures
        )
    
    def _set_up_result_dicts(self) -> None:
        self._results = dict()
        self._figures = dict()

    def _characterise_background_tcr_distribution(self) -> None:
        background_subsample = self._background_data.sample(n=10_000, random_state=420)
        background_embeddings = self._model_computation_cacher.calc_vector_representations(background_subsample)

        print(background_subsample.head())

        pca = PCA()
        embeddings_pca = pca.fit_transform(background_embeddings)

        scatter_figure = self._generate_2d_scatter(embeddings_pca)
        scatter_by_bv = self._generate_2d_scatter_with_colors(embeddings_pca, background_subsample.TRBV)
        scatter_by_bv = self._generate_2d_scatter_with_colors(embeddings_pca, background_subsample.TRBJ)
        scatter_by_bv = self._generate_2d_scatter_with_colors(embeddings_pca, background_subsample.TRAV)
        scatter_by_bv = self._generate_2d_scatter_with_colors(embeddings_pca, background_subsample.TRAJ)
        scatter_by_bv = self._generate_2d_scatter_with_colorscale(embeddings_pca, background_subsample.CDR3B.str.len())
        scatter_by_bv = self._generate_2d_scatter_with_colorscale(embeddings_pca, background_subsample.beta_pgen.map(lambda x: -np.log10(x)))

        pca_summary_figure = self._generate_pca_summary_figure(pca)

        self._figures["scatter"] = scatter_figure
        self._figures["scatter_by_bv"] = scatter_by_bv
        self._figures["pca_summary"] = pca_summary_figure
    
    @staticmethod
    def _generate_2d_scatter(data: ndarray) -> Figure:
        fig, axes = plt.subplots()
        axes.scatter(data[:,0], data[:,1])
        return fig
    
    @staticmethod
    def _generate_2d_scatter_with_colors(data: ndarray, color_labels: Series) -> Figure:
        fig, axes = plt.subplots()

        unique_labels = color_labels.unique()
        for label in unique_labels:
            data_for_label = data[color_labels == label]
            axes.scatter(data_for_label[:,0], data_for_label[:,1])
        
        return fig
    
    @staticmethod
    def _generate_2d_scatter_with_colorscale(data: ndarray, scale: Series) -> Figure:
        fig, axes = plt.subplots()
        axes.scatter(data[:,0], data[:,1], c=scale)
        return fig
    
    @staticmethod
    def _generate_pca_summary_figure(pca: PCA) -> Figure:
        fig, axes = plt.subplots()
        axes.bar(range(pca.n_features_), pca.explained_variance_ratio_)
        return fig