from abc import ABC, abstractmethod
from pandas import DataFrame
from pathlib import Path

from src.model.tcr_metric import TcrMetric
from src.model_analyser.model_computation_cacher import ModelComputationCacher
from src.model_analyser.analysis_result import AnalysisResult

from typing import Dict

class Analysis(ABC):
    def __init__(
        self,
        background_data: DataFrame,
        background_pgen: DataFrame,
        labelled_data: Dict[str, DataFrame],
        tcr_model: TcrMetric,
        working_directory: Path,
    ) -> None:
        self._background_data = background_data
        self._background_pgen = background_pgen
        self._labelled_data = labelled_data
        self._model = tcr_model
        self._model_computation_cacher = ModelComputationCacher(
            tcr_model, working_directory
        )

    @abstractmethod
    def run(self) -> AnalysisResult:
        pass
