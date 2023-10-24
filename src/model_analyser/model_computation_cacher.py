import hashlib
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from pathlib import Path
import pickle
from typing import Callable, IO

from src.model.tcr_metric import TcrMetric
from src.model.tcr_representation_model import TcrRepresentationModel
from src.model_analyser.tcr_edit_distance_records.tcr_edit_distance_record_collection import (
    TcrEditDistanceRecordCollection,
)


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

    def calc_vector_representations(self, tcrs: DataFrame) -> ndarray:
        if isinstance(self._model, TcrRepresentationModel):
            representation_model: TcrRepresentationModel = self._model
        else:
            raise RuntimeError(f"{self._model.name} is not a")

        argument_hash_str = self._get_argument_hash_str(tcrs)
        filename = f"reps_{argument_hash_str}.npy"

        representation_model: TcrRepresentationModel = self._model
        compute_fn = lambda: representation_model.calc_vector_representations(tcrs)

        vector_representations = self.get_cached_or_compute_array(filename, compute_fn)

        return vector_representations

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

    def get_tcr_edit_record_collection(self) -> TcrEditDistanceRecordCollection:
        save_path = self._cache_dir / "tcr_edit_record_collection_state.pkl"

        if save_path.is_file():
            with open(save_path, "rb") as f:
                state_dict = pickle.load(f)
            return TcrEditDistanceRecordCollection.from_state_dict(state_dict)

        return TcrEditDistanceRecordCollection()

    def save_tcr_edit_record_collection(
        self, tcr_edit_record_collection: TcrEditDistanceRecordCollection
    ) -> None:
        save_path = self._cache_dir / "tcr_edit_record_collection_state.pkl"

        with open(save_path, "wb") as f:
            tcr_edit_record_collection.save(f)

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
