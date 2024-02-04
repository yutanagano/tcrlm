import math
import pandas as pd
from pandas import DataFrame
from src.model_analyser.tcr_edit_distance_records.tcr_edit import JunctionEdit, Residue, Position, Chain
from typing import Iterable, List, Literal, Tuple


def get_all_tcr_variants(tcr: DataFrame) -> DataFrame:
    junction_variants = get_junction_variants(tcr)
    return junction_variants
    

def get_junction_variants(tcr: DataFrame) -> DataFrame:
    alpha_junction_variants = _get_chain_specific_junction_variants(tcr, Chain.Alpha)
    beta_junction_variants = _get_chain_specific_junction_variants(tcr, Chain.Beta)
    return pd.concat([alpha_junction_variants, beta_junction_variants])


def _get_chain_specific_junction_variants(tcr: DataFrame, chain: Chain) -> DataFrame:
    if chain == Chain.Alpha:
        junction_type = "CDR3A"
    elif chain == Chain.Beta:
        junction_type = "CDR3B"
    else:
        raise ValueError

    edits = []
    edited_tcrs = []

    junction = tcr[junction_type].item()
    max_junction_length_post_edit = len(junction) + 1

    for edit_index in range(max_junction_length_post_edit):
        insertions, junctions_with_insertion = _get_insertions_and_variants_at_index(junction, edit_index, chain)
        tcrs_with_insertion = [
            _get_tcr_with_edited_component(tcr, junction_type, junction_with_insertion)
            for junction_with_insertion in junctions_with_insertion
        ]

        edits.extend(insertions)
        edited_tcrs.extend(tcrs_with_insertion)

        if edit_index < len(junction):
            deletion, junction_with_deletion = _get_deletion_and_variant_at_index(junction, edit_index, chain)
            tcr_with_deletion = _get_tcr_with_edited_component(tcr, junction_type, junction_with_deletion)

            substitutions, junctions_with_substitution = _get_subs_and_variants_at_index(junction, edit_index, chain)
            tcrs_with_substitution = [
                _get_tcr_with_edited_component(tcr, junction_type, junction_with_substitution)
                for junction_with_substitution in junctions_with_substitution
            ]

            edits.append(deletion)
            edited_tcrs.append(tcr_with_deletion)

            edits.extend(substitutions)
            edited_tcrs.extend(tcrs_with_substitution)

    edited_tcrs = pd.concat(edited_tcrs)
    edited_tcrs["edit"] = edits

    return edited_tcrs


def _get_insertions_and_variants_at_index(junction: str, insertion_index: int, chain: Chain) -> Tuple[Iterable[JunctionEdit], Iterable[str]]:
    insertions = []
    junctions_with_insertion = []

    insertion_position = _get_edit_position_for_junction_from_edit_index(junction, insertion_index)

    for residue_to_insert in Residue:
        if residue_to_insert == Residue.null:
            continue

        insertion = JunctionEdit(chain, insertion_position, Residue.null, residue_to_insert)
        junction_with_insertion = (
            junction[:insertion_index]
            + residue_to_insert.name
            + junction[insertion_index:]
        )

        insertions.append(insertion)
        junctions_with_insertion.append(junction_with_insertion)

    return (insertions, junctions_with_insertion)


def _get_deletion_and_variant_at_index(junction: str, deletion_index: int, chain: Chain) -> Tuple[JunctionEdit, str]:
    deletion_position = _get_edit_position_for_junction_from_edit_index(junction, deletion_index)
    residue_to_delete = Residue[junction[deletion_index]]

    deletion = JunctionEdit(chain, deletion_position, residue_to_delete, Residue.null)
    junction_with_deletion = (
        junction[:deletion_index] + junction[deletion_index + 1 :]
    )

    return (deletion, junction_with_deletion)


def _get_subs_and_variants_at_index(junction: str, substitution_index: int, chain: Chain) -> Tuple[Iterable[JunctionEdit], Iterable[str]]:
    substitutions = []
    junctions_with_substitution = []

    substitution_position = _get_edit_position_for_junction_from_edit_index(
        junction, substitution_index
    )
    from_residue = Residue[junction[substitution_index]]

    for to_residue in Residue:
        if to_residue in (from_residue, Residue.null):
            continue

        substitution = JunctionEdit(chain, substitution_position, from_residue, to_residue)
        junction_with_substitution = (
            junction[:substitution_index]
            + to_residue.name
            + junction[substitution_index + 1 :]
        )

        substitutions.append(substitution)
        junctions_with_substitution.append(junction_with_substitution)

    return (substitutions, junctions_with_substitution)


def _get_edit_position_for_junction_from_edit_index(junction: str, edit_index: int) -> Position:
    position_enum_index = math.ceil(edit_index / len(junction) * 5)
    if position_enum_index == 0:
        position_enum_index = 1 # automatic enum indexing starts at one
    return Position(position_enum_index)


def _get_tcr_with_edited_component(tcr: DataFrame, component_name: str, new_component_value) -> DataFrame:
    tcr_with_edited_component = tcr.copy(deep=True)
    tcr_with_edited_component[component_name] = new_component_value
    return tcr_with_edited_component