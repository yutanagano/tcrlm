from src.model_analyser.tcr_edit_distance_records.coverage_summary import (
    CoverageSummary,
)
from src.model_analyser.tcr_edit_distance_records.tcr_edit_distance_record import (
    TcrEditDistanceRecord,
)
from src.model_analyser.tcr_edit_distance_records import tcr_edit
from src.model_analyser.tcr_edit_distance_records.tcr_edit import JunctionEdit

from itertools import permutations, product
import pickle
from typing import Dict, Iterable, List


class TcrEditDistanceRecordCollection:
    MARGINAL_NUM_ESTIMATES_REQUIRED = 100

    def __init__(self) -> None:
        self.initialise_edit_record_dictionary()

    def initialise_edit_record_dictionary(self) -> None:
        self.edit_record_dictionary: Dict[JunctionEdit, TcrEditDistanceRecord] = dict()

        for edit in tcr_edit.get_all_tcr_edits():
            self.edit_record_dictionary[edit] = TcrEditDistanceRecord()

    def update_edit_record(self, edit: JunctionEdit, distance: float):
        relevant_edit_record = self.edit_record_dictionary[edit]
        relevant_edit_record.add_distance_sample(distance)

    def print_current_estimation_coverage(self):
        print("Number of estimates at positions:")
        print(self.get_coverage_summary_over_junction_positions())

        print("Number of estimates for each edit:")
        print(self.get_coverage_summary_over_aa_indelsubs())

    def get_coverage_summary_over_junction_positions(self) -> CoverageSummary:
        return CoverageSummary(self.get_num_estimates_over_junction_positions())

    def get_coverage_summary_over_aa_indelsubs(self) -> CoverageSummary:
        return CoverageSummary(self.get_num_estimates_over_aa_indelsubs())

    def has_sufficient_coverage(self) -> bool:
        return (
            self.has_sufficient_junction_coverage()
            and self.has_sufficient_v_gene_coverage()
        )

    def has_sufficient_junction_coverage(self) -> bool:
        has_sufficient_coverage_over_junction_positions = all(
            num_estimates_made >= self.MARGINAL_NUM_ESTIMATES_REQUIRED
            for num_estimates_made in self.get_num_estimates_over_junction_positions()
        )

        has_sufficient_coverage_over_aa_indelsubs = all(
            num_estimates_made >= self.MARGINAL_NUM_ESTIMATES_REQUIRED
            for num_estimates_made in self.get_num_estimates_over_aa_indelsubs()
        )

        return has_sufficient_coverage_over_junction_positions and has_sufficient_coverage_over_aa_indelsubs

    def get_num_estimates_over_junction_positions(self) -> List[int]:
        return [
            self.get_num_estimates_at_junction_position(chain, position)
            for chain, position in product(tcr_edit.Chain, tcr_edit.Position)
        ]

    def get_num_estimates_at_junction_position(self, chain: str, position: str) -> int:
        edits_at_position = [
            edit for edit in self.edit_record_dictionary if
            edit.is_on_chain(chain) and edit.is_at_position(position)
        ]
        return self.get_num_estimates_accross_specified_edits(edits_at_position)

    def get_num_estimates_over_aa_indelsubs(self) -> List[int]:
        all_ordered_pairs_of_distinct_residues = permutations(tcr_edit.Residue, r=2)
        return [
            self.get_num_estimates_for_aa_indelsub(from_residue, to_residue)
            for from_residue, to_residue in all_ordered_pairs_of_distinct_residues
        ]

    def get_num_estimates_for_aa_indelsub(
        self, from_residue: str, to_residue: str
    ) -> int:
        relevant_indelsubs = [
            edit
            for edit in self.edit_record_dictionary
            if edit.is_from(from_residue) and edit.is_to(to_residue)
        ]
        return self.get_num_estimates_accross_specified_edits(relevant_indelsubs)

    def get_num_estimates_accross_specified_edits(self, edits: Iterable) -> int:
        relevant_edit_records = [self.edit_record_dictionary[edit] for edit in edits]
        return sum(
            [edit_record.num_distances_sampled for edit_record in relevant_edit_records]
        )

    def has_sufficient_v_gene_coverage(self) -> bool:
        # TODO
        return True
    
    def has_sufficient_central_sub_coverage(self) -> bool:
        central_positions = [tcr_edit.Position.M1, tcr_edit.Position.M2, tcr_edit.Position.M3]
        num_estimates_over_central_positions = [
            self.get_num_estimates_for_subs_at_junction_position(chain, position)
            for chain, position in product(tcr_edit.Chain, central_positions)
        ]
        return all(
            num_estimates_made >= 10_000
            for num_estimates_made in num_estimates_over_central_positions
        )
    
    def get_num_estimates_for_subs_at_junction_position(self, chain: str, position: str) -> int:
        edits_at_position = [
            edit for edit in self.edit_record_dictionary if
            edit.is_on_chain(chain) and edit.is_at_position(position) and not edit.is_from(tcr_edit.Residue.null) and not edit.is_to(tcr_edit.Residue.null)
        ]
        return self.get_num_estimates_accross_specified_edits(edits_at_position)

    def save(self, f) -> None:
        state_dict = self.get_state_dict()
        pickle.dump(state_dict, f)

    def get_state_dict(self) -> dict:
        state_dict = {
            str(edit): edit_record.get_state_dict()
            for edit, edit_record in self.edit_record_dictionary.items()
        }
        return state_dict

    @staticmethod
    def from_save(f) -> "TcrEditDistanceRecordCollection":
        state_dict = pickle.load(f)

        return TcrEditDistanceRecordCollection.from_state_dict(state_dict)

    @staticmethod
    def from_state_dict(state_dict: dict) -> "TcrEditDistanceRecordCollection":
        edit_record_collection = TcrEditDistanceRecordCollection()

        for edit_str, edit_record_state_dict in state_dict.items():
            edit = JunctionEdit.from_str(edit_str)
            edit_record = TcrEditDistanceRecord.from_state_dict(edit_record_state_dict)

            edit_record_collection.edit_record_dictionary[edit] = edit_record

        return edit_record_collection
