from dataclasses import dataclass
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np
from ring import algorithms
from ring import base
from ring import maths
from ring import ml
from ring import sim2real
from ring import utils

from diodem import load_data
from diodem._src import _is_arm_or_gait


@cache
def _load_sys(exp_id: int) -> base.System:
    return base.System.from_xml(
        Path(__file__).parent.joinpath(f"xmls/{_is_arm_or_gait(exp_id)}.xml")
    )


@cache
def _get_sys(exp_id, anchor: str, include_links: tuple[str]):
    sys = _load_sys(exp_id).morph_system(new_anchor=anchor)
    delete = list(set(sys.link_names) - set(include_links))
    return sys.delete_system(delete, strict=False)


def _max_coords_after_omc_pos_offset(sys: base.System, data: dict) -> dict:

    data_out = dict()
    for link_name, max_cord in zip(sys.link_names, sys.omc):
        if max_cord is None:
            continue
        cs_name, marker, pos_offset = (
            max_cord.coordinate_system_name,
            max_cord.pos_marker_number,
            max_cord.pos_marker_constant_offset,
        )
        pos = data[cs_name][f"marker{marker}"]
        quat = data[cs_name]["quat"]
        pos_with_offset = pos + maths.rotate(pos_offset, quat)
        data_out[link_name] = dict(pos=pos_with_offset, quat=quat)

    return data_out


@dataclass
class IMTP:
    segments: Optional[list[str]] = None
    mag: bool = False
    flex: bool = False
    sparse: bool = False
    joint_axes_1d: bool = False
    joint_axes_1d_field: bool = True
    joint_axes_2d: bool = False
    joint_axes_2d_field: bool = False
    dof: bool = False
    dof_field: bool = False
    hz: float = 100.0
    dt: bool = True
    model_name_suffix: Optional[str] = None
    # we divide by these factors
    scale_acc: float = 1.0
    scale_gyr: float = 1.0
    scale_mag: float = 1.0
    scale_dt: float = 1.0

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    def imus(self) -> list[str]:
        segs = self.segments
        if self.sparse:
            segs = segs[0:1] + [segs[-1]]
        return [f"imu{seg[-1]}" for seg in segs]

    def sys(self, exp_id: int) -> base.System:
        include_links = self.segments + self.imus()
        sys = _get_sys(exp_id, self.segments[0], tuple(include_links))
        sys = sys.change_model_name(suffix=f"_{len(self.segments)}Seg")
        return sys

    def sys_noimu(self, exp_id: int) -> tuple[base.System, dict[str, str]]:
        sys_noimu, imu_attachment = self.sys(exp_id).make_sys_noimu()
        assert tuple(sys_noimu.link_parents) == self.lam
        assert sys_noimu.link_names == self.segments
        return sys_noimu, imu_attachment

    def getDOF(self, exp_id: int) -> dict[str, int]:
        sys = self.sys_noimu(exp_id)[0]
        dofs = {
            name: base.QD_WIDTHS[joint_type]
            for name, joint_type in zip(sys.link_names, sys.link_types)
        }
        return dofs

    def getJointAxes2d(self, exp_id: int, seg: str) -> np.ndarray:
        "returns (6,) array"
        # only exp_id > 6
        assert exp_id >= 6
        assert seg in ["seg2", "seg3"]
        return np.array([0, 1, 0, 0, 0, 1.0])

    @property
    def lam(self) -> tuple:
        return tuple(range(-1, len(self.segments) - 1))

    @property
    def N(self):
        return len(self.segments)

    def getSlices(self) -> dict[str, slice]:
        _, slices = self._get_F_and_slices()
        return slices

    def getF(self) -> int:
        F, _ = self._get_F_and_slices()
        return F

    def _get_F_and_slices(self):
        F = 6
        slices = {"acc": slice(0, 3), "gyr": slice(3, 6)}
        if self.mag:
            slices["mag"] = slice(6, 9)
            F += 3
        if self.joint_axes_1d_field:
            slices["ja_1d"] = slice(F, F + 3)
            F += 3
        if self.joint_axes_2d_field:
            slices["ja_2d"] = slice(F, F + 6)
            F += 6
        if self.dof_field:
            slices["dof"] = slice(F, F + 3)
            F += 3
        if self.dt:
            slices["dt"] = slice(F, F + 1)
            F += 1
        return F, slices

    def name(self, exp_id: int, motion_start: str, motion_stop: Optional[str] = None):
        if motion_stop is None:
            motion_stop = ""
        model_name = (
            self.sys(exp_id).change_model_name(suffix=self.model_name_suffix).model_name
        )
        flex, mag, ja_1d, ja_2d, dof = (
            int(self.flex),
            int(self.mag),
            int(self.joint_axes_1d),
            int(self.joint_axes_2d),
            int(self.dof),
        )
        return (
            f"{model_name}_exp{str(exp_id).rjust(2, '0')}_{motion_start}_{motion_stop}"
            + f"_flex_{flex}_mag_{mag}_ja1d_{ja_1d}_ja2d_{ja_2d}_dof_{dof}"
        )


def _build_Xy_xs_xsnoimu(
    exp_id: int, motion_start: str, motion_stop: str | None, imtp: IMTP
) -> tuple[np.ndarray, np.ndarray, base.Transform, base.Transform]:

    data = load_data(exp_id, motion_start, motion_stop, resample_to_hz=imtp.hz)
    sys = imtp.sys(exp_id)
    sys_noimu, imu_attachment = imtp.sys_noimu(exp_id)

    max_coords = _max_coords_after_omc_pos_offset(sys, data)
    xs = sim2real.xs_from_raw(
        sys,
        max_coords,
        qinv=True,
    )
    xs_noimu = sim2real.xs_from_raw(
        sys_noimu,
        max_coords,
        qinv=True,
    )

    T = xs.shape()
    N, F = imtp.N, imtp.getF()

    X, y = np.zeros((T, N, F)), np.zeros((T, N, 4))

    slices = imtp.getSlices()

    imu_key = "imu_nonrigid" if imtp.flex else "imu_rigid"
    for i, seg in enumerate(imtp.segments):
        if seg in list(imu_attachment.values()):
            X_seg = data[seg][imu_key]
            X[:, i, slices["acc"]] = X_seg["acc"] / imtp.scale_acc
            X[:, i, slices["gyr"]] = X_seg["gyr"] / imtp.scale_gyr
            if imtp.mag:
                X[:, i, slices["mag"]] = X_seg["mag"] / imtp.scale_mag

    DOFs = imtp.getDOF(exp_id)
    if imtp.joint_axes_1d:
        X_joint_axes = algorithms.joint_axes(sys_noimu, xs, sys)
        for i, seg in enumerate(imtp.segments):
            if DOFs[seg] == 1:
                X[:, i, slices["ja_1d"]] = X_joint_axes[seg]["joint_axes"]  # noqa: E203

    if imtp.joint_axes_2d:
        for i, seg in enumerate(imtp.segments):
            if DOFs[seg] == 2:
                ja_2d = imtp.getJointAxes2d(exp_id, seg)
                X[:, i, slices["ja_2d"]] = ja_2d[None]

    if imtp.dof:
        for i, seg in enumerate(imtp.segments):
            dof_seg = DOFs[seg]
            if dof_seg in [1, 2, 3]:
                one_hot_array = np.zeros((3,))
                one_hot_array[dof_seg - 1] = 1.0
                X[:, i, slices["dof"]] = one_hot_array[None]

    if imtp.dt:
        for i, seg in enumerate(imtp.segments):
            X[:, i, slices["dt"]] = (1 / imtp.hz) / imtp.scale_dt

    y_dict = algorithms.rel_pose(sys_noimu, xs, sys)
    y_rootfull = algorithms.sensors.root_full(sys_noimu, xs, sys, child_to_parent=True)
    y_dict = utils.dict_union(y_dict, y_rootfull)
    for i, seg in enumerate(imtp.segments):
        y[:, i] = y_dict[seg]

    return X, y, xs, xs_noimu


_mae_metrices = dict(
    mae_deg=lambda q, qhat: jnp.rad2deg(jnp.mean(maths.angle_error(q, qhat)[:, 2500:])),
    incl_deg=lambda q, qhat: jnp.rad2deg(
        jnp.mean(maths.inclination_loss(q, qhat)[:, 2500:])
    ),
)


def benchmark(
    imtp: IMTP,
    exp_id: int,
    motion_start: str,
    filter: Optional[ml.AbstractFilter] = None,
    motion_stop: Optional[str] = None,
    warmup: float = 0.0,
    return_cb: bool = False,
    cb_metrices: dict = _mae_metrices,
):
    "`warmup` is in seconds."

    X, y, xs, xs_noimu = _build_Xy_xs_xsnoimu(exp_id, motion_start, motion_stop, imtp)

    if filter is None:
        return X, y, xs, xs_noimu

    if return_cb:
        return ml.callbacks.EvalXyTrainingLoopCallback(
            filter,
            cb_metrices,
            X,
            y,
            imtp.lam,
            imtp.name(exp_id, motion_start, motion_stop),
            link_names=imtp.segments,
        )

    yhat, _ = filter.apply(X=X, y=y, lam=imtp.lam)

    warmup = int(warmup * imtp.hz)
    errors = dict()
    for i, seg in enumerate(imtp.segments):
        ae = np.rad2deg(maths.angle_error(y[:, i], yhat[:, i])[warmup:])
        errors[seg] = {}
        errors[seg]["mae"] = np.mean(ae)
        errors[seg]["std"] = np.std(ae)

    return errors, X, y, yhat, xs, xs_noimu
