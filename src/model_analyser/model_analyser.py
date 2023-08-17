import pandas as pd
from pathlib import Path
import re
from typing import Optional, Type

from src.model.tcr_metric import TcrMetric
from src.model_analyser.analysis import (
    Analysis,
    KnnAnalysis,
    PgenAnalysis,
    PrecisionRecallAnalysis,
)
from src.model_analyser.analysis_result import AnalysisResult


BACKGROUND_DATA_PATH = "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/tanno/test.csv"
BACKGROUND_PGEN_PATH = "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/tanno/test_pgens.csv"
LABELLED_DATA_PATHS = {
    "gdb_holdout": "/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/gdb/test.csv",
}


class ModelAnalyser:
    def __init__(self, working_directory: Optional[Path] = None) -> None:
        self._set_working_directory(working_directory)
        self._load_data()

    def _set_working_directory(self, working_directory: Optional[Path]) -> None:
        if working_directory is not None:
            self._working_directory = working_directory
        else:
            self._working_directory = Path.cwd()

    def _load_data(self) -> None:
        self._background_data = pd.read_csv(BACKGROUND_DATA_PATH)
        self._background_pgen = pd.read_csv(BACKGROUND_PGEN_PATH)
        self._labelled_data = {
            name: pd.read_csv(path) for name, path in LABELLED_DATA_PATHS.items()
        }

    def analyse(self, tcr_model: TcrMetric) -> None:
        analyses = [KnnAnalysis, PrecisionRecallAnalysis, PgenAnalysis]

        analysis_results = [
            self._run_analysis(AnalysisClass, tcr_model) for AnalysisClass in analyses
        ]

        path_to_save_directory = self._get_path_to_save_directory(tcr_model)

        for analysis_result in analysis_results:
            analysis_result.save(path_to_save_directory)

    def _run_analysis(
        self, analysis_class: Type[Analysis], tcr_model: TcrMetric
    ) -> AnalysisResult:
        print(f"Running {analysis_class.__name__}...")

        analysis = analysis_class(
            background_data=self._background_data,
            background_pgen=self._background_pgen,
            labelled_data=self._labelled_data,
            tcr_model=tcr_model,
            working_directory=self._working_directory,
        )
        analysis_result = analysis.run()
        return analysis_result

    def _get_path_to_save_directory(self, tcr_model: TcrMetric) -> Path:
        save_parent_dir = self._get_save_parent_dir()
        mode_name_without_special_characeters = self._remove_special_characters(
            tcr_model.name
        )

        model_save_dir = save_parent_dir / mode_name_without_special_characeters
        model_save_dir.mkdir(exist_ok=True)

        return model_save_dir

    def _get_save_parent_dir(self) -> Path:
        save_parent_dir = self._working_directory / "benchmarks"
        save_parent_dir.mkdir(exist_ok=True)
        return save_parent_dir

    def _remove_special_characters(self, s: str) -> str:
        without_whitespace = re.sub(r"\s+", "_", s)
        non_alphanumerics_removed = re.sub(r"\W", "", without_whitespace)
        return non_alphanumerics_removed
