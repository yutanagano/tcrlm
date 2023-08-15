from src.model_benchmarker import ModelBenchmarker
from src.model.tcr_metric import BetaCdr3Levenshtein


def test_benchmark(tmp_path):
    benchmarker = ModelBenchmarker(working_directory=tmp_path)
    model = BetaCdr3Levenshtein()

    benchmarker.benchmark(model)

    expected_results_save_dir = tmp_path/"benchmarks"/"Beta_CDR3_Levenshtein"

    assert expected_results_save_dir.is_dir()