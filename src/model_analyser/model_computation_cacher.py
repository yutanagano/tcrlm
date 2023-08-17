import hashlib
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from pathlib import Path
from typing import Callable, IO

from src.model.tcr_metric import TcrMetric


class ModelComputationCacher:
    def __init__(self, model: TcrMetric, working_directory: Path) -> None:
        self._model = model
        self._cache_dir = self._get_path_to_cache_dir(working_directory)

    def _get_path_to_cache_dir(self, working_directory: Path) -> Path:
        cache_dir = working_directory / ".model_computation_cache"
        cache_dir.mkdir(exist_ok=True)

        model_cache_dir = cache_dir / self._model.name
        model_cache_dir.mkdir(exist_ok=True)

        return model_cache_dir

    def calc_cdist_matrix(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame
    ) -> ndarray:
        argument_hash_str = self._get_argument_hash_str(anchor_tcrs, comparison_tcrs)
        filename = f"cdist_{argument_hash_str}.npy"
        compute_fn = lambda: self._model.calc_cdist_matrix(anchor_tcrs, comparison_tcrs)

        cdist_matrix = self.get_cached_or_compute_array(filename, compute_fn)

        return cdist_matrix

    def calc_pdist_vector(self, tcrs: DataFrame) -> ndarray:
        argument_hash_str = self._get_argument_hash_str(tcrs)
        filename = f"pdist_{argument_hash_str}.npy"
        compute_fn = lambda: self._model.calc_pdist_vector(tcrs)

        pdist_vector = self.get_cached_or_compute_array(filename, compute_fn)

        return pdist_vector

    def get_cached_or_compute_array(
        self, filename: str, compute_fn: Callable
    ) -> ndarray:
        file = self._cache_dir / filename

        if file.is_file():
            return np.load(file)
        else:
            computed_result = compute_fn()
            np.save(
                file,
                computed_result,
            )
            return computed_result

    def _get_argument_hash_str(self, *args) -> str:
        stringified_args = [str(arg).encode("utf-8") for arg in args]
        hashed_args = [hashlib.sha256(arg).hexdigest() for arg in stringified_args]
        return "_".join(hashed_args)

    def get_readable_buffer(self, filename: str) -> IO:
        file = self._cache_dir / filename
        file.touch()
        return open(file, "r")

    def get_appendable_buffer(self, filename: str) -> IO:
        file = self._cache_dir / filename
        return open(file, "a")
