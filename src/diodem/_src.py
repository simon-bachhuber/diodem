from functools import cache
from typing import Optional

import numpy as np
import pandas as pd
import tree_utils

from diodem import dataverse_github
from diodem import utils


@cache
def _is_arm_or_gait(exp_id: int) -> str:
    exp_id = str(exp_id).rjust(2, "0")
    search = lambda arm_or_gait: dataverse_github.listdir(
        f"dataset/{arm_or_gait}/exp{exp_id}"
    )
    if len(search("arm")) > 0:
        return "arm"
    elif len(search("gait")) > 0:
        return "gait"
    else:
        raise Exception(f"`exp_id`={exp_id} was not found in repo.")


def _path_up_to_motion(exp_id: int) -> str:
    return f"dataset/{_is_arm_or_gait(exp_id)}/exp{str(exp_id).rjust(2, '0')}"


@cache
def _load_timings(exp_id: int) -> list[str]:
    omc_files = dataverse_github.listdir(
        filter_prefix=_path_up_to_motion(exp_id),
        filter_suffix="omc.csv",
    )
    motions = [file.split("/")[3] for file in omc_files]
    motions.sort(key=lambda ele: int(ele[6:8]))
    return motions


def _stack_from_df(df: pd.DataFrame, prefix: str, wxyz: str):
    cols = [prefix + ele for ele in wxyz]
    arr = []
    for col in cols:
        arr.append(df[col].to_numpy()[:, None])
        assert arr[-1].ndim == 2
    return np.concatenate(arr, axis=1)


@cache
def _load_data(exp_id: int, motion: str):
    path = (
        f"{_path_up_to_motion(exp_id)}/{motion}/exp{str(exp_id).rjust(2, '0')}"
        f"_{motion[:8]}_"
    )

    downloader = lambda file: dataverse_github.download(path + file)

    omc = pd.read_csv(downloader("omc.csv"), delimiter=",", skiprows=2)
    omc_hz = int(open(downloader("omc.csv")).readline().split(":")[1].lstrip().rstrip())
    imu_rigid = pd.read_csv(downloader("imu_rigid.csv"), delimiter=",", skiprows=2)
    imu_rigid_hz = int(
        open(downloader("imu_rigid.csv")).readline().split(":")[1].lstrip().rstrip()
    )
    imu_nonrigid = pd.read_csv(
        downloader("imu_nonrigid.csv"), delimiter=",", skiprows=2
    )
    imu_nonrigid_hz = int(
        open(downloader("imu_nonrigid.csv")).readline().split(":")[1].lstrip().rstrip()
    )
    assert imu_rigid_hz == imu_nonrigid_hz

    data = {}
    for seg in range(1, 6):
        data_seg = {}
        seg = f"seg{seg}"
        data[seg] = data_seg

        # quat
        data_seg["quat"] = _stack_from_df(omc, seg + "_quat_", "wxyz")

        # markers
        for marker in range(1, 5):
            marker = f"marker{marker}"
            data_seg[marker] = _stack_from_df(omc, seg + "_" + marker + "_", "xyz")

        # imu
        for imu_name, imu in zip(
            ["imu_rigid", "imu_nonrigid"], [imu_rigid, imu_nonrigid]
        ):
            data_seg_imu = {}
            data_seg[imu_name] = data_seg_imu
            for accgyrmag in ["acc", "gyr", "mag"]:
                data_seg_imu[accgyrmag] = _stack_from_df(
                    imu, seg + "_" + accgyrmag + "_", "xyz"
                )

    return data, omc_hz, imu_rigid_hz


def _convert_motion(exp_id: int, motion: str | int) -> str:
    timings = _load_timings(exp_id)
    for timing in timings:
        if isinstance(motion, str):
            if timing[9:] == motion:
                break
        else:
            if int(timing[6:8]) == motion:
                break
    else:
        raise Exception(f"motion `{motion}` not in {timings}")

    return timing


def load_data(
    exp_id: int,
    motion_start: str | int = 1,
    motion_stop: Optional[str | int] = None,
    resample_to_hz: float = 100.0,
) -> dict:

    timings = _load_timings(exp_id)
    motion_start = _convert_motion(exp_id, motion_start)
    assert motion_start in timings

    if motion_stop is None:
        motion_stop = motion_start
    elif motion_stop == -1:
        motion_stop = timings[-1]
    else:
        motion_stop = _convert_motion(exp_id, motion_stop)
        assert motion_stop in timings

    motion_start_i = timings.index(motion_start)
    motion_stop_i = timings.index(motion_stop)
    assert motion_start_i <= motion_stop_i, "Empty sequence, stop < start"

    motions = timings[motion_start_i : (motion_stop_i + 1)]  # noqa: E203
    data = []
    for motion in motions:
        data_motion, hz_omc, hz_imu = _load_data(exp_id, motion)
        data.append(data_motion)

    data = tree_utils.tree_batch(data, along_existing_first_axis=True, backend="numpy")

    data = utils.resample(
        data,
        hz_in=utils.hz_helper(
            data.keys(),
            imus=["imu_rigid", "imu_nonrigid"],
            hz_imu=hz_imu,
            hz_omc=hz_omc,
        ),
        hz_out=resample_to_hz,
        vecinterp_method="cubic",
    )
    data = utils.crop_tail(data, resample_to_hz, strict=True, verbose=False)

    return data
