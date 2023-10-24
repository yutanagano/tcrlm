from collections import deque
import pickle
import pytest

from src.model_analyser.tcr_edit_distance_records.tcr_edit import (
    TcrEdit,
    Position,
    Residue,
)
from src.model_analyser.tcr_edit_distance_records import tcr_edit
from src.model_analyser.tcr_edit_distance_records.tcr_edit_distance_record_collection import (
    TcrEditDistanceRecordCollection,
)


@pytest.fixture
def tcr_edit_record_collection():
    return TcrEditDistanceRecordCollection()


def sufficiently_fill_and_return(record_collection) -> TcrEditDistanceRecordCollection:
    for edit in tcr_edit.get_all_tcr_edits():
        for _ in range(20):
            record_collection.update_edit_record(edit, 1.0)

    return record_collection


def test_update_edit_record(
    tcr_edit_record_collection: TcrEditDistanceRecordCollection,
):
    edit = TcrEdit(Position.M2, Residue.A, Residue.C)

    tcr_edit_record_collection.update_edit_record(edit, 1.0)

    edit_record = tcr_edit_record_collection.edit_record_dictionary[edit]

    assert edit_record.distance_sample == deque([1.0])
    assert edit_record.num_distances_sampled == 1


def test_print_current_estimation_coverage(
    tcr_edit_record_collection: TcrEditDistanceRecordCollection, capfd
):
    tcr_edit_record_collection.print_current_estimation_coverage()

    out, _ = capfd.readouterr()

    assert (
        out == "Number of estimates at positions:\n"
        "min: 0, max: 0, mean: 0.0\n"
        "Number of estimates for each edit:\n"
        "min: 0, max: 0, mean: 0.0\n"
    )


def test_has_sufficient_coverage(
    tcr_edit_record_collection: TcrEditDistanceRecordCollection,
):
    filled_edit_record_collection = sufficiently_fill_and_return(
        tcr_edit_record_collection
    )

    assert filled_edit_record_collection.has_sufficient_coverage()


def test_save_load(
    tcr_edit_record_collection: TcrEditDistanceRecordCollection, tmp_path
):
    edit = TcrEdit(Position.M2, Residue.A, Residue.C)
    tcr_edit_record_collection.update_edit_record(edit, 1.0)

    with open(tmp_path / "save", "wb") as f:
        tcr_edit_record_collection.save(f)

    with open(tmp_path / "save", "rb") as f:
        new_edit_record_collection = TcrEditDistanceRecordCollection.from_save(f)

    reloaded_edit_record = new_edit_record_collection.edit_record_dictionary[edit]

    assert reloaded_edit_record.distance_sample == deque([1.0])
    assert reloaded_edit_record.num_distances_sampled == 1
