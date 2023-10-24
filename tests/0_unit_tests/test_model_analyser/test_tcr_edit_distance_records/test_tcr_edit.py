from src.model_analyser.tcr_edit_distance_records import tcr_edit
from src.model_analyser.tcr_edit_distance_records.tcr_edit import (
    TcrEdit,
    Position,
    Residue,
)


def test_is_at():
    tcr_edit = TcrEdit(Position.M2, Residue.A, Residue.C)

    assert tcr_edit.is_at(Position.M2)


def test_is_central():
    central_tcr_edit = TcrEdit(Position.M2, Residue.A, Residue.C)
    flank_tcr_edit = TcrEdit(Position.C_TERM, Residue.A, Residue.C)

    assert central_tcr_edit.is_central
    assert not flank_tcr_edit.is_central


def test_is_from():
    tcr_edit = TcrEdit(Position.M2, Residue.A, Residue.C)

    assert tcr_edit.is_from(Residue.A)


def test_is_to():
    tcr_edit = TcrEdit(Position.M2, Residue.A, Residue.C)

    assert tcr_edit.is_to(Residue.C)


def test_hash():
    tcr_edit = TcrEdit(Position.M2, Residue.A, Residue.C)

    assert hash(tcr_edit) == hash((Position.M2, Residue.A, Residue.C))


def test_eq():
    anchor_tcr_edit = TcrEdit(Position.M2, Residue.A, Residue.C)
    positive_comparison = TcrEdit(Position.M2, Residue.A, Residue.C)
    negative_comparison = TcrEdit(Position.C_TERM, Residue.A, Residue.C)

    assert anchor_tcr_edit == positive_comparison
    assert anchor_tcr_edit != negative_comparison


def test_repr():
    tcr_edit = TcrEdit(Position.M2, Residue.A, Residue.C)

    assert repr(tcr_edit) == "M2.A.C"
    assert str(tcr_edit) == "M2.A.C"


def test_from_str():
    expected = TcrEdit(Position.M2, Residue.A, Residue.C)
    from_str = TcrEdit.from_str("M2.A.C")

    assert from_str == expected


def test_get_all_tcr_edits():
    all_tcr_edits = tcr_edit.get_all_tcr_edits()

    num_positions = 5
    num_residues = 21

    assert len(all_tcr_edits) == num_residues * (num_residues - 1) * num_positions
