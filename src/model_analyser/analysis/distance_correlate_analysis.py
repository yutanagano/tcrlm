import math
import pandas as pd
from pandas import DataFrame
from typing import Iterable, Tuple
from src.model_analyser.analysis import Analysis
from src.model_analyser.analysis_result import AnalysisResult
from src.model_analyser.tcr_edit_distance_records.tcr_edit_distance_record_collection import (
    TcrEditDistanceRecordCollection,
)
from src.model_analyser.tcr_edit_distance_records.tcr_edit import (
    Residue,
    Position,
    JunctionEdit,
)
from src.model_analyser.tcr_edit_distance_records.tcr_edit_distance_record_collection_analyser import (
    TcrEditDistanceRecordCollectionAnalyser,
)
from src.model_analyser.tcr_edit_distance_records import tcr_edit_generator
from src.model.tcr_representation_model import Sceptr


class DistanceCorrelateAnalysis(Analysis):
    def run(self) -> AnalysisResult:
        if self._current_model_is_irrelevant_for_this_analysis():
            return AnalysisResult("na")

        edit_record_collection = self._get_edit_record_collection()
        filled_edit_record_collection = (
            self._fill_edit_record_collection_with_estimates(edit_record_collection)
        )
        self._save_edit_record_collection(filled_edit_record_collection)

        analyser = TcrEditDistanceRecordCollectionAnalyser(
            filled_edit_record_collection
        )

        results = dict()
        results["edit_distance_summary"] = analyser.make_summary_dict()

        figures = dict()
        figures["edit_distance_summary"] = analyser.make_summary_figure()
        figures[
            "substitution_distance_vs_blosum"
        ] = analyser.make_substitution_distance_vs_blosum_figure()

        return AnalysisResult(
            "distance_correlate_analysis", results=results, figures=figures
        )

    def _current_model_is_irrelevant_for_this_analysis(self) -> bool:
        return not isinstance(self._model, Sceptr)

    def _get_edit_record_collection(self) -> TcrEditDistanceRecordCollection:
        return self._model_computation_cacher.get_tcr_edit_record_collection()

    def _save_edit_record_collection(
        self, tcr_edit_record_collection: TcrEditDistanceRecordCollection
    ) -> None:
        self._model_computation_cacher.save_tcr_edit_record_collection(
            tcr_edit_record_collection
        )

    def _fill_edit_record_collection_with_estimates(
        self, edit_record_collection: TcrEditDistanceRecordCollection
    ) -> TcrEditDistanceRecordCollection:
        num_tcrs_processed = 0

        while not edit_record_collection.has_sufficient_coverage():
            tcr = self._get_random_background_tcr_as_df()
            edits_and_resulting_distances = (
                self._get_all_possible_edits_and_resulting_distances(tcr)
            )

            for edit, distance in edits_and_resulting_distances:
                edit_record_collection.update_edit_record(edit, distance)

            num_tcrs_processed += 1

            if (num_tcrs_processed % 10) == 0:
                print(f"{num_tcrs_processed} TCRs processed...")
                edit_record_collection.print_current_estimation_coverage()

        return edit_record_collection

    def _get_random_background_tcr_as_df(self) -> DataFrame:
        return self._background_data.sample(n=1)

    def _get_all_possible_edits_and_resulting_distances(
        self, tcr: DataFrame
    ) -> Iterable[Tuple[JunctionEdit, float]]:
        original_tcr = tcr
        tcr_variants = tcr_edit_generator.get_all_tcr_variants(tcr)
        distances_between_original_and_edited_tcrs = self._model.calc_cdist_matrix(
            original_tcr, tcr_variants
        ).squeeze()
        return zip(tcr_variants["edit"], distances_between_original_and_edited_tcrs)