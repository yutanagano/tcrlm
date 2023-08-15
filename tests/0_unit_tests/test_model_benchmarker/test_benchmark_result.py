import json
from matplotlib.figure import Figure
from pathlib import Path

from src.model_benchmarker.benchmark_result import BenchmarkResult


def test_save(tmp_path):
    BENCHMARK_RESULT_NAME = "foobar"
    RESUTLS_DICT = {"foo": "bar"}
    FIGURE_NAME = "foo"

    benchmark_result = BenchmarkResult(
        name=BENCHMARK_RESULT_NAME,
        results_dict=RESUTLS_DICT,
        figures={FIGURE_NAME: Figure()}
    )

    benchmark_result.save(tmp_path)

    assert results_dict_saved_as_json(tmp_path/BENCHMARK_RESULT_NAME, RESUTLS_DICT)
    assert figures_saved_as_png(tmp_path/BENCHMARK_RESULT_NAME, FIGURE_NAME)

def results_dict_saved_as_json(save_dir: Path, expected_dict: dict) -> bool:
    with open(save_dir/"results.json", "r") as f:
        results_dict = json.load(f)
    
    return results_dict == expected_dict

def figures_saved_as_png(save_dir: Path, expected_figure_name: str) -> bool:
    expected_png_file = save_dir/f"{expected_figure_name}.png"
    return expected_png_file.is_file()