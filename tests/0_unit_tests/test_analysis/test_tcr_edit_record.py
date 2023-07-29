from collections import deque
import pytest

from src.analysis.tcr_edit_record import TcrEditRecord


@pytest.fixture
def tcr_edit_record():
    tcr_edit_record = TcrEditRecord()

    for distance in range(20_000):
        tcr_edit_record.add_distance_estimate(distance)

    return tcr_edit_record

def test_add_distance_estimate(tcr_edit_record):
    assert tcr_edit_record.distance_estimates == deque(range(10_000, 20_000))
    assert tcr_edit_record.num_estimates_made == 20_000


def test_average_distance(tcr_edit_record):
    assert tcr_edit_record.average_distance == 14999.5

def test_min_distance(tcr_edit_record):
    assert tcr_edit_record.min_distance == 10_000

def test_max_distance(tcr_edit_record):
    assert tcr_edit_record.max_distance == 19_999

def test_get_state_dict(tcr_edit_record):
    expected = {
        "distance_estimates": deque(range(10_000, 20_000)),
        "num_estimates_made": 20_000
    }

    assert tcr_edit_record.get_state_dict() == expected

def test_from_state_dict():
    new_tcr_edit_record = TcrEditRecord.from_state_dict(
        {
            "distance_estimates": deque(range(10_000, 20_000)),
            "num_estimates_made": 20_000
        }
    )

    assert new_tcr_edit_record.distance_estimates == deque(range(10_000, 20_000))
    assert new_tcr_edit_record.num_estimates_made == 20_000