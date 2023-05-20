"""
Base class for benchmarking pure metrics.
"""


from .benchmarking_pipeline import BenchmarkingPipeline
from pathlib import Path


class PureMetricBenchmarkingPipeline(BenchmarkingPipeline):
    MODEL_NAME: str

    def run_from_clargs(cls) -> None:
        cls.main()

    def main(cls) -> None:
        cls.setup(None)
        for ds_name, ds_df in cls.labelled_data.items():
            cls.becnhmark_on_labelled_data(ds_name, ds_df)
        cls.save()
        print("Done!")

    def instantiate_model(cls, model_save_dir: Path) -> None:
        cls.model = type("", (), {"name": cls.MODEL_NAME})