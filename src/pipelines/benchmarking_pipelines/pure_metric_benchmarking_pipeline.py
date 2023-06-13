"""
Base class for benchmarking pure metrics.
"""


from .benchmarking_pipeline import BenchmarkingPipeline
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from pathlib import Path
from tqdm import tqdm


class PureMetricBenchmarkingPipeline(BenchmarkingPipeline):
    MODEL_NAME: str

    def run_from_clargs(cls) -> None:
        cls.main()

    def main(cls) -> None:
        cls.setup(None)
        cls.evaluate_pgen_vs_representation_space_density()
        for ds_name, ds_df in cls.labelled_data.items():
            cls.becnhmark_on_labelled_data(ds_name, ds_df)
        cls.save()
        print("Done!")

    def instantiate_model(cls, model_save_dir: Path) -> None:
        cls.model = type("", (), {"name": cls.MODEL_NAME})

    def get_avg_dist_to_100nn_over_background(cls) -> ndarray:
        avg_dists = []

        for i in tqdm(range(0, len(cls.background_data), 1000)):
            dists_batch = cls.get_cdist(
                cls.background_data.iloc[i:i+1000],
                cls.background_data
            ).squeeze()

            for dists in dists_batch:
                dists_to_closest_100 = np.partition(dists, kth=100)[:100]
                avg_dist = dists_to_closest_100.mean()
                avg_dists.append(avg_dist)

        return np.array(avg_dists, dtype=np.float32)
    
    def get_cdist(cls, ds_a_df: DataFrame, ds_b_df: DataFrame) -> ndarray:
        raise NotImplementedError()
    
    def evaluate_svm_performance(cls, ds_df: DataFrame) -> dict:
        return dict()