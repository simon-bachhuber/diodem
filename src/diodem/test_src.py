import pytest

from diodem import load_data


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
