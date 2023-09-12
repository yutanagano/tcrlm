import math
import pandas as pd
from pandas import DataFrame
from typing import Iterable, Tuple

from src.model_analyser.analysis import Analysis
from src.model_analyser.analysis_result import AnalysisResult
from src.model_analyser.tcr_edit_distance_records.tcr_edit_distance_record_collection import TcrEditDistanceRecordCollection
from src.model_analyser.tcr_edit_distance_records.tcr_edit import Residue, Position, TcrEdit
from src.model_analyser.tcr_edit_distance_records.tcr_edit_distance_record_collection_analyser import TcrEditDistanceRecordCollectionAnalyser


class DistanceCorrelateAnalysis(Analysis):
    def run(self) -> AnalysisResult:
        edit_record_collection = self._get_edit_record_collection()
        filled_edit_record_collection = self._fill_edit_record_collection_with_estimates(edit_record_collection)
        self._save_edit_record_collection(filled_edit_record_collection)

        analyser = TcrEditDistanceRecordCollectionAnalyser(filled_edit_record_collection)

        figures = dict()
        figures["beta_junction_insertion_distances"] = analyser.make_beta_junction_insertion_figure()
        figures["beta_junction_deletion_distances"] = analyser.make_beta_junction_deletion_figure()
        figures["beta_junction_substitution_distances"] = analyser.make_beta_junction_substitution_figure()
        figures["substitution_distance_vs_blosum"] = analyser.make_substitution_distance_vs_blosum_figure()

        return AnalysisResult("distance_correlate_analysis", figures=figures)

    def _get_edit_record_collection(self) -> TcrEditDistanceRecordCollection:
        return self._model_computation_cacher.get_tcr_edit_record_collection()
    
    def _save_edit_record_collection(self, tcr_edit_record_collection: TcrEditDistanceRecordCollection) -> None:
        self._model_computation_cacher.save_tcr_edit_record_collection(tcr_edit_record_collection)
    
    def _fill_edit_record_collection_with_estimates(self, edit_record_collection: TcrEditDistanceRecordCollection) -> TcrEditDistanceRecordCollection:
        num_tcrs_processed = 0

        while not edit_record_collection.has_sufficient_coverage():
            tcr = self._get_random_background_tcr_as_df()
            edits_and_resulting_distances = self._get_all_possible_edits_and_resulting_distances(tcr)

            for edit, distance in edits_and_resulting_distances:
                edit_record_collection.update_edit_record(edit, distance)

            num_tcrs_processed += 1

            if (num_tcrs_processed % 10) == 0:
                print(f"{num_tcrs_processed} TCRs processed...")
                edit_record_collection.print_current_estimation_coverage()

        return edit_record_collection
    
    def _get_random_background_tcr_as_df(self) -> DataFrame:
        return self._background_data.sample(n=1)
    
    def _get_all_possible_edits_and_resulting_distances(self, tcr: DataFrame) -> Iterable[Tuple[TcrEdit, float]]:
        original_tcr = tcr
        edits, edited_tcrs = self._get_edits_and_resulting_variants_of_tcr(tcr)
        distances_between_original_and_edited_tcrs = self._model.calc_cdist_matrix(original_tcr, edited_tcrs).squeeze()
        return zip(edits, distances_between_original_and_edited_tcrs)
    
    def _get_edits_and_resulting_variants_of_tcr(self, tcr: DataFrame) -> Tuple[Iterable[TcrEdit], DataFrame]:
        trav_edits, tcrs_with_trav_edits = self._get_all_trav_edits_and_resulting_tcrs(tcr)
        trbv_edits, tcrs_with_trbv_edits = self._get_all_trbv_edits_and_resulting_tcrs(tcr)
        alpha_junction_edits, tcrs_with_alpha_junction_edits = self._get_all_alpha_junction_edits_and_resulting_tcrs(tcr)
        beta_junction_edits, tcrs_with_beta_junction_edits = self._get_all_beta_junction_edits_and_resulting_tcrs(tcr)
        
        all_edits = [*trav_edits, *trbv_edits, *alpha_junction_edits, *beta_junction_edits]
        all_edited_tcrs = [*tcrs_with_trav_edits, *tcrs_with_trbv_edits, *tcrs_with_alpha_junction_edits, *tcrs_with_beta_junction_edits]
        all_edited_tcrs_as_df = pd.concat(all_edited_tcrs, axis="index")

        return (all_edits, all_edited_tcrs_as_df)
    
    def _get_all_trav_edits_and_resulting_tcrs(self, tcr: DataFrame) -> Tuple[Iterable[TcrEdit], Iterable[DataFrame]]:
        #TODO
        return ([], [])
    
    def _get_all_trbv_edits_and_resulting_tcrs(self, tcr: DataFrame) -> Tuple[Iterable[TcrEdit], Iterable[DataFrame]]:
        #TODO
        return ([], [])
    
    def _get_all_alpha_junction_edits_and_resulting_tcrs(self, tcr: DataFrame) -> Tuple[Iterable[TcrEdit], Iterable[DataFrame]]:
        #TODO
        return ([], [])
    
    def _get_all_beta_junction_edits_and_resulting_tcrs(self, tcr: DataFrame) -> Tuple[Iterable[TcrEdit], Iterable[DataFrame]]:
        edits = []
        edited_tcrs = []
        
        junction = tcr.CDR3B.item()
        max_junction_length_post_edit = len(junction) + 1

        for edit_index in range(max_junction_length_post_edit):
            insertions, junctions_with_insertion = self._get_aa_insertions_at_index_and_resulting_junctions(junction, edit_index)
            tcrs_with_insertion = [self._get_tcr_with_edited_component(tcr, "CDR3B", junction_with_insertion) for junction_with_insertion in junctions_with_insertion]

            edits.extend(insertions)
            edited_tcrs.extend(tcrs_with_insertion)

            if edit_index < len(junction):
                deletions, junctions_with_deletion = self._get_aa_deletions_at_index_and_resulting_junctions(junction, edit_index)
                tcrs_with_deletion = [self._get_tcr_with_edited_component(tcr, "CDR3B", junction_with_deletion) for junction_with_deletion in junctions_with_deletion]

                substitutions, junctions_with_substitution = self._get_aa_substitutions_at_index_and_resulting_junctions(junction, edit_index)
                tcrs_with_substitution = [self._get_tcr_with_edited_component(tcr, "CDR3B", junction_with_substitution) for junction_with_substitution in junctions_with_substitution]

                edits.extend(deletions)
                edited_tcrs.extend(tcrs_with_deletion)
                edits.extend(substitutions)
                edited_tcrs.extend(tcrs_with_substitution)

        return (edits, edited_tcrs)
    
    def _get_aa_insertions_at_index_and_resulting_junctions(self, junction: str, insertion_index: int) -> Tuple[Iterable[TcrEdit], Iterable[str]]:
        insertions = []
        junctions_with_insertion = []

        insertion_position = self._get_edit_position_for_junction_from_edit_index(junction, insertion_index)

        for residue_to_insert in Residue:
            if residue_to_insert == Residue.null:
                continue

            insertion = TcrEdit(insertion_position, Residue.null, residue_to_insert)
            junction_with_insertion = junction[:insertion_index] + residue_to_insert.name + junction[insertion_index:]
            
            insertions.append(insertion)
            junctions_with_insertion.append(junction_with_insertion)
        
        return (insertions, junctions_with_insertion)
    
    def _get_aa_deletions_at_index_and_resulting_junctions(self, junction: str, deletion_index: int) -> Tuple[Iterable[TcrEdit], Iterable[str]]:
        deletion_position = self._get_edit_position_for_junction_from_edit_index(junction, deletion_index)
        residue_to_delete = Residue[junction[deletion_index]]

        deletion = TcrEdit(deletion_position, residue_to_delete, Residue.null)
        junction_with_deletion = junction[:deletion_index] + junction[deletion_index+1:]
        
        return ([deletion], [junction_with_deletion])
    
    def _get_aa_substitutions_at_index_and_resulting_junctions(self, junction: str, substitution_index: int) -> Tuple[Iterable[TcrEdit], Iterable[str]]:
        substitutions = []
        junctions_with_substitution = []

        substitution_position = self._get_edit_position_for_junction_from_edit_index(junction, substitution_index)
        from_residue = Residue[junction[substitution_index]]

        for to_residue in Residue:
            if to_residue in (from_residue, Residue.null):
                continue

            substitution = TcrEdit(substitution_position, from_residue, to_residue)
            junction_with_substitution = junction[:substitution_index] + to_residue.name + junction[substitution_index+1:]
            
            substitutions.append(substitution)
            junctions_with_substitution.append(junction_with_substitution)
        
        return (substitutions, junctions_with_substitution)
    
    def _get_edit_position_for_junction_from_edit_index(self, junction: str, edit_index: int) -> Position:
        position_enum_index = math.ceil(edit_index / len(junction) * 5)
        if position_enum_index == 0:
            position_enum_index = 1
        return Position(position_enum_index)
    
    def _get_tcr_with_edited_component(self, tcr: DataFrame, component_name: str, new_component_value) -> DataFrame:
        tcr_with_edited_component = tcr.copy(deep=True)
        tcr_with_edited_component[component_name] = new_component_value
        return tcr_with_edited_component