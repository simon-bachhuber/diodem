import pytest

from diodem import load_all_valid_motions_in_trial
from diodem import load_data
from diodem import load_timing_relative_to_complete_trial


def test_src():
    # problem: exp doesnt exist
    with pytest.raises(Exception):
        load_data(0)

    load_data(1)

    # problem: motion doesnt exist
    with pytest.raises(Exception):
        load_data(2, 17)

    # problem: stop < start
    with pytest.raises(AssertionError):
        load_data(2, 4, 3)

    load_data(1, 1, 4)

    load_data(1, "pause1")

    # problem: stop < start
    with pytest.raises(AssertionError):
        load_data(1, "pause1", "canonical")

    load_data(1, "pause1", "pause2")


def test_dataverse():
    load_data(9, backend="dataverse")


def test_github():
    load_data(10, backend="github")


def test_load_motions():
    load_all_valid_motions_in_trial(1)


def test_load_timings():
    load_timing_relative_to_complete_trial(1, "slow1")
