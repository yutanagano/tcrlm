import blosum
import itertools
from itertools import chain
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import math
import numpy as np
import random
from typing import Iterable, List, Optional, Set

from src.model_analyser.tcr_edit_distance_records.tcr_edit_distance_record_collection import (
    TcrEditDistanceRecordCollection,
)
from src.model_analyser.tcr_edit_distance_records.tcr_edit import (
    Position,
    Residue,
    TcrEdit,
)
from src.model_analyser.tcr_edit_distance_records.tcr_edit_distance_record import (
    TcrEditDistanceRecord,
)


class TcrEditDistanceRecordCollectionAnalyser:
    def __init__(
        self, tcr_edit_record_collection: TcrEditDistanceRecordCollection
    ) -> None:
        self.edit_record_collection = tcr_edit_record_collection

    def make_summary_dict(self) -> dict:
        insertion_distances = self._get_insertion_distances_over_positions()
        deletion_distances = self._get_deletion_distances_over_positions()
        substitution_distances = self._get_substitution_distances_over_positions()

        def get_distance_std_dict(mean_distances, standard_deviations) -> dict:
            return {
                "mean_distances": mean_distances,
                "standard_deviations": standard_deviations
            }

        summary = dict()

        summary["insertions"] = get_distance_std_dict(*insertion_distances)
        summary["deletions"] = get_distance_std_dict(*deletion_distances)
        summary["substitutions"] = get_distance_std_dict(*substitution_distances)

        return summary

    def make_summary_figure(self) -> Figure:
        insertion_distances = self._get_insertion_distances_over_positions()
        deletion_distances = self._get_deletion_distances_over_positions()
        substitution_distances = self._get_substitution_distances_over_positions()

        violin_position_array = np.arange(len(Position)) * 6

        fig, ax = plt.subplots()

        ax.errorbar(violin_position_array, insertion_distances[0], yerr=insertion_distances[1], fmt="o")
        ax.errorbar(violin_position_array+1, deletion_distances[0], yerr=deletion_distances[1], fmt="o")
        ax.errorbar(violin_position_array+2, substitution_distances[0], yerr=substitution_distances[1], fmt="o")
        ax.set_xticks(violin_position_array+1, [position.name for position in Position])

        ax.set_ylim(0)

        ax.set_ylabel("distance")
        ax.set_xlabel("CDR3 region")

        return fig

    def _get_insertion_distances_over_positions(self) -> List[List[float]]:
        all_insertions = self._get_all_junction_aa_insertions()
        insertions_over_positions = [
            edits.intersection(all_insertions)
            for edits in self._get_all_junction_edits_over_positions()
        ]
        distances_over_positions = [
            self._get_mean_std_distance_from_specified_edits(edits)
            for edits in insertions_over_positions
        ]

        return list(zip(*distances_over_positions))

    def _get_deletion_distances_over_positions(self) -> List[List[float]]:
        all_deletions = self._get_all_junction_aa_deletions()
        deletions_over_positions = [
            edits.intersection(all_deletions)
            for edits in self._get_all_junction_edits_over_positions()
        ]
        distances_over_positions = [
            self._get_mean_std_distance_from_specified_edits(edits)
            for edits in deletions_over_positions
        ]

        return list(zip(*distances_over_positions))

    def _get_substitution_distances_over_positions(self) -> List[List[float]]:
        all_substitutions = self._get_all_junction_aa_substitutions()
        substitutions_over_positions = [
            edits.intersection(all_substitutions)
            for edits in self._get_all_junction_edits_over_positions()
        ]
        distances_over_positions = [
            self._get_mean_std_distance_from_specified_edits(edits)
            for edits in substitutions_over_positions
        ]

        return list(zip(*distances_over_positions))

    def _generate_distance_violinplot_over_junction_positions(
        self, distances_over_junction_positions: List[Iterable[float]], title: str
    ) -> Figure:
        fig, ax = plt.subplots()

        ax.violinplot(distances_over_junction_positions, positions=range(len(Position)))
        ax.set_xticks(range(len(Position)), [position.name for position in Position])
        ax.set_title(title)

        return fig

    def make_substitution_distance_vs_blosum_figure(self) -> Figure:
        blosum_similarities = []
        model_distances = []

        for from_residue, to_residue in itertools.permutations(Residue, r=2):
            blosum_similarity = blosum.BLOSUM(62)[from_residue.name][to_residue.name]
            model_distance = self._get_average_distance_for_central_substitution(
                from_residue, to_residue
            )

            estimation_impossible_due_to_missing_data = model_distance is None
            if estimation_impossible_due_to_missing_data:
                continue

            blosum_similarities.append(blosum_similarity)
            model_distances.append(model_distance)

        fig, ax = plt.subplots()

        ax.scatter(blosum_similarities, model_distances, alpha=0.2)
        ax.set_xlabel("BLOSUM62 similarity")
        ax.set_ylabel("Model distance")

        return fig

    def _get_all_junction_aa_insertions(self) -> Set[TcrEdit]:
        return {
            edit
            for edit in self.edit_record_collection.edit_record_dictionary
            if edit.is_from(Residue.null)
        }

    def _get_all_junction_aa_deletions(self) -> Set[TcrEdit]:
        return {
            edit
            for edit in self.edit_record_collection.edit_record_dictionary
            if edit.is_to(Residue.null)
        }

    def _get_all_junction_aa_substitutions(self) -> Set[TcrEdit]:
        return {
            edit
            for edit in self.edit_record_collection.edit_record_dictionary
            if not (edit.is_from(Residue.null) or edit.is_to(Residue.null))
        }

    def _get_all_junction_edits_over_positions(self) -> List[Set[TcrEdit]]:
        return [self._get_all_edits_at_position(position) for position in Position]

    def _get_all_edits_at_position(self, position: Position) -> Set[TcrEdit]:
        return {
            edit
            for edit in self.edit_record_collection.edit_record_dictionary
            if edit.is_at(position)
        }

    def _get_distance_sample_from_specified_edits(
        self, edits: Iterable[TcrEdit]
    ) -> List:
        edit_records = [
            self.edit_record_collection.edit_record_dictionary[edit] for edit in edits
        ]
        distance_samples = [edit_record.distance_sample for edit_record in edit_records]

        if not any([edit_record.is_overfilled for edit_record in edit_records]):
            return list(chain.from_iterable(distance_samples))

        sample_weights = self._get_sample_weights(edit_records)
        num_distances_to_keep_from_each_sample = [
            round(sample_weight * TcrEditDistanceRecord.DISTANCE_SAMPLES_CAPACITY)
            for sample_weight in sample_weights
        ]
        weighted_distance_samples = [
            random.sample(distances, num_to_sample)
            for distances, num_to_sample in zip(
                distance_samples, num_distances_to_keep_from_each_sample
            )
        ]

        return list(chain.from_iterable(weighted_distance_samples))
    
    def _get_mean_std_distance_from_specified_edits(self, edits: Iterable[TcrEdit]) -> tuple:
        edit_records = [
            self.edit_record_collection.edit_record_dictionary[edit] for edit in edits
        ]
        mean_distances = [edit_record.average_distance for edit_record in edit_records]
        var_distances = [edit_record.var_distance for edit_record in edit_records]
        num_samples = [edit_record.num_distances_sampled for edit_record in edit_records]
        weights = [num / sum(num_samples) for num in num_samples]

        mean_distance = sum([weight * mean_distance for (weight, mean_distance) in zip(weights, mean_distances)])
        var_distance = sum([weight * var for (weight, var) in zip(weights, var_distances)])

        return (mean_distance, math.sqrt(var_distance))

    def _get_sample_weights(
        self, edit_records: Iterable[TcrEditDistanceRecord]
    ) -> List[float]:
        num_distances_sampled_per_edit_record = [
            edit_record.num_distances_sampled for edit_record in edit_records
        ]
        total_num_distances_sampled = sum(num_distances_sampled_per_edit_record)
        return [
            num_distances_sampled / total_num_distances_sampled
            for num_distances_sampled in num_distances_sampled_per_edit_record
        ]

    def _get_average_distance_for_central_substitution(
        self, from_residue: Residue, to_residue: Residue
    ) -> Optional[float]:
        relevant_edits = [
            edit
            for edit in self.edit_record_collection.edit_record_dictionary
            if edit.is_from(from_residue) and edit.is_to(to_residue) and edit.is_central
        ]

        edit_records_per_position = [
            self.edit_record_collection.edit_record_dictionary[edit]
            for edit in relevant_edits
        ]
        edit_records_with_data = [
            edit_record
            for edit_record in edit_records_per_position
            if edit_record.num_distances_sampled > 0
        ]
        distance_per_available_position = [
            edit_record.average_distance for edit_record in edit_records_with_data
        ]

        if len(distance_per_available_position) == 0:
            return None

        return sum(distance_per_available_position) / len(
            distance_per_available_position
        )
