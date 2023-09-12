from collections import deque
import pytest

from src.model_analyser.tcr_edit_distance_records.tcr_edit_distance_record import TcrEditDistanceRecord


@pytest.fixture
def tcr_edit_record():
    tcr_edit_record = TcrEditDistanceRecord()

    for distance in range(20_000):
        tcr_edit_record.add_distance_sample(distance)

    return tcr_edit_record


def test_add_distance_estimate(tcr_edit_record):
    assert tcr_edit_record.distance_sample == deque(range(10_000, 20_000))
    assert tcr_edit_record.num_distances_sampled == 20_000


def test_average_distance(tcr_edit_record):
    assert tcr_edit_record.average_distance == 14999.5


def test_min_distance(tcr_edit_record):
    assert tcr_edit_record.min_distance == 10_000


def test_max_distance(tcr_edit_record):
    assert tcr_edit_record.max_distance == 19_999


def test_get_state_dict(tcr_edit_record):
    expected = {
        "distance_sample": deque(range(10_000, 20_000)),
        "num_distances_sampled": 20_000,
    }

    assert tcr_edit_record.get_state_dict() == expected


def test_from_state_dict():
    new_tcr_edit_record = TcrEditDistanceRecord.from_state_dict(
        {
            "distance_sample": deque(range(10_000, 20_000)),
            "num_distances_sampled": 20_000,
        }
    )

    assert new_tcr_edit_record.distance_sample == deque(range(10_000, 20_000))
    assert new_tcr_edit_record.num_distances_sampled == 20_000
