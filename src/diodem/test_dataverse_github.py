import pytest

from diodem import dataverse_github


def test_download_from_github_repo():
    path_on_disk = dataverse_github.download(
        path_in_repo="README.md", path_to_cache="~/.diodem_cache", backend="github"
    )

    with pytest.raises(Exception):
        # typo -> file not found
        path_on_disk = dataverse_github.download(
            path_in_repo="README.m",
            path_to_cache="~/.diodem_cache",
            backend="github",
        )

    assert path_on_disk.exists() and path_on_disk.is_file()


def test_download_from_dataverse_repo():
    path_on_disk = dataverse_github.download(
        path_in_repo="images/KC_gait_nobackground.png",
        path_to_cache="~/.diodem_cache",
        backend="dataverse",
    )

    with pytest.raises(Exception):
        # typo -> file not found
        path_on_disk = dataverse_github.download(
            path_in_repo="images/KC_gait_nobackground.pn",
            path_to_cache="~/.diodem_cache",
            backend="dataverse",
        )

    assert path_on_disk.exists() and path_on_disk.is_file()
