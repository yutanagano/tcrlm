from src.model_benchmarker.benchmark_result import BenchmarkResult


class BenchmarkResultChecker:
    def __init__(self, benchmark_result: BenchmarkResult) -> None:
        self._benchmark_result = benchmark_result

    def name_is(self, name: str) -> bool:
        return self._benchmark_result.name == name
    
    def has_results_dict(self) -> bool:
        return self._benchmark_result._results_dict is not None
    
    def has_figures(self) -> bool:
        return self._benchmark_result._figures is not None