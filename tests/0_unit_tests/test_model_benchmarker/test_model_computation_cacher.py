import hashlib
import numpy as np
from pandas import DataFrame

from src.model_benchmarker.model_computation_cacher import ModelComputationCacher
from src.model.tcr_metric import BetaCdr3Levenshtein


def test_calc_cdist_matrix(tmp_path, mock_data_df):
    model = BetaCdr3Levenshtein()
    cacher = ModelComputationCacher(model, tmp_path)
    mock_data_hashed = hash_df(mock_data_df)

    result = cacher.calc_cdist_matrix(mock_data_df, mock_data_df)

    expected = np.array(
        [
            [0,4,0],
            [4,0,4],
            [0,4,0]
        ]
    )
    expected_cache_file_name = f"cdist_{mock_data_hashed}_{mock_data_hashed}.npy"
    expected_cache_path = tmp_path/".model_computation_cache"/model.name/expected_cache_file_name

    assert np.array_equal(result, expected)
    assert expected_cache_path.is_file()

def test_calc_pdist_vector(tmp_path, mock_data_df):
    model = BetaCdr3Levenshtein()
    cacher = ModelComputationCacher(model, tmp_path)
    mock_data_hashed = hash_df(mock_data_df)

    result = cacher.calc_pdist_vector(mock_data_df)

    expected = np.array([4,0,4])
    expected_cache_file_name = f"pdist_{mock_data_hashed}.npy"
    expected_cache_path = tmp_path/".model_computation_cache"/model.name/expected_cache_file_name

    assert np.array_equal(result, expected)
    assert expected_cache_path.is_file()

def test_get_cached_or_compute_array(tmp_path):
    FILENAME = "planted.npy"

    model = BetaCdr3Levenshtein()
    cacher = ModelComputationCacher(model, tmp_path)

    cache_dir = tmp_path/".model_computation_cache"/model.name

    planted = np.array([1,2,3])
    np.save(cache_dir/FILENAME, planted)

    result = cacher.get_cached_or_compute_array(FILENAME, lambda: np.zeros(3))

    assert np.array_equal(result, planted)

def hash_df(df: DataFrame) -> str:
    stringified_df = str(df).encode("utf-8")
    hashed_df = hashlib.sha256(stringified_df)
    return hashed_df.hexdigest()