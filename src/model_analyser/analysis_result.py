import json
from matplotlib.figure import Figure
from pathlib import Path
from typing import Dict, Optional


class AnalysisResult:
    def __init__(
        self,
        name: str,
        results_dict: Optional[Dict] = None,
        figures: Optional[Dict[str, Figure]] = None,
    ) -> None:
        self.name = name
        self._results_dict = results_dict
        self._figures = figures

    def save(self, save_parent_dir: Path) -> None:
        save_dir = self._create_save_directory(save_parent_dir)
        self._save_results_dict(save_dir)
        self._save_figures(save_dir)

    def _create_save_directory(self, save_parent_dir: Path) -> Path:
        save_dir = save_parent_dir / self.name
        save_dir.mkdir(exist_ok=True)
        return save_dir

    def _save_results_dict(self, save_dir: Path) -> None:
        if self._results_dict is None:
            return

        with open(save_dir / "results.json", "w") as f:
            json.dump(self._results_dict, f, indent=4)

    def _save_figures(self, save_dir: Path) -> None:
        if self._figures is None:
            return

        for filename, figure in self._figures.items():
            figure.savefig(save_dir / filename)
