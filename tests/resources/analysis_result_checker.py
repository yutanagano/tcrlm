from src.model_analyser.analysis_result import AnalysisResult


class AnalysisResultChecker:
    def __init__(self, benchmark_result: AnalysisResult) -> None:
        self._benchmark_result = benchmark_result

    def name_is(self, name: str) -> bool:
        return self._benchmark_result.name == name

    def has_results_dict(self) -> bool:
        return self._benchmark_result._results_dict is not None

    def has_figures(self) -> bool:
        return self._benchmark_result._figures is not None
