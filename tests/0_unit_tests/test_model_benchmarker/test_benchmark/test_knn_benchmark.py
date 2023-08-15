from src.model_benchmarker.benchmark import KnnBenchmark

from tests.resources.benchmark_result_checker import BenchmarkResultChecker


def test_run(mock_bg_data, mock_pgens, mock_labelled_data_dict, beta_cdr3_levenshtein_model, tmp_path):
    benchmark = KnnBenchmark(
        background_data=mock_bg_data,
        background_pgen=mock_pgens,
        labelled_data=mock_labelled_data_dict,
        tcr_model=beta_cdr3_levenshtein_model,
        working_directory=tmp_path
    )

    result = benchmark.run()
    result_checker = BenchmarkResultChecker(result)

    assert result_checker.name_is("knn")
    assert result_checker.has_results_dict()
    assert not result_checker.has_figures()