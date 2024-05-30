import pytest

from diodem import dataverse_github


def test_download_from_github_repo():
    path_on_disk = dataverse_github.download("imgs/pose1_noBG.png", backend="github")

    with pytest.raises(Exception):
        # typo -> file not found
        path_on_disk = dataverse_github.download("imgs/pose1_noBG.pn", backend="github")

    assert path_on_disk.exists() and path_on_disk.is_file()


def test_download_from_dataverse_repo():
    path_on_disk = dataverse_github.download(
        "images/KC_gait_nobackground.png", backend="dataverse"
    )

    with pytest.raises(Exception):
        # typo -> file not found
        path_on_disk = dataverse_github.download(
            "images/KC_gait_nobackground.pn", backend="dataverse"
        )

    assert path_on_disk.exists() and path_on_disk.is_file()
