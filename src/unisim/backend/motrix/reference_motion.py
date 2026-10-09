"""Cold-path Motrix forward kinematics for reference-motion materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from unisim.backend.base import SimBackend

if TYPE_CHECKING:
    import motrixsim as mtx


@dataclass(frozen=True)
class MotrixReferenceKinematics:
    """Reference link states in UniLab's wxyz quaternion contract."""

    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray


def materialize_reference_kinematics(
    backend: SimBackend,
    *,
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    root_lin_vel: np.ndarray,
    root_ang_vel: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    body_names: tuple[str, ...],
) -> MotrixReferenceKinematics:
    """Run batched FK once using the already materialized Motrix scene model.

    This adapter is deliberately kept outside the environment package.  It is
    only called during environment initialization; no asset/model inspection
    occurs in ``step`` or ``reset``.
    """
    if backend.backend_type != "motrix":
        raise ValueError("SONIC raw reference materialization currently requires Motrix")

    import motrixsim as mtx

    model: mtx.SceneModel = backend._model  # pyright: ignore[reportPrivateUsage]
    count = int(root_pos.shape[0])
    expected = {
        "root_quat_xyzw": (count, 4),
        "root_lin_vel": (count, 3),
        "root_ang_vel": (count, 3),
        "joint_pos": (count, len(model.joint_names)),
        "joint_vel": (count, len(model.joint_names)),
    }
    values = {
        "root_quat_xyzw": root_quat_xyzw,
        "root_lin_vel": root_lin_vel,
        "root_ang_vel": root_ang_vel,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
    }
    if root_pos.shape != (count, 3):
        raise ValueError(f"root_pos must have shape {(count, 3)}, got {root_pos.shape}")
    for name, shape in expected.items():
        if values[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {values[name].shape}")

    qpos = np.concatenate((root_pos, root_quat_xyzw, joint_pos), axis=-1).astype(
        np.float32, copy=False
    )
    qvel = np.concatenate((root_lin_vel, np.zeros_like(root_ang_vel), joint_vel), axis=-1).astype(
        np.float32, copy=False
    )
    data = mtx.SceneData(model, batch=[count])
    data.set_dof_pos(np.ascontiguousarray(qpos), model)
    data.set_dof_vel(np.ascontiguousarray(qvel))
    model.floating_bases[0].set_global_angular_velocity(
        data, np.ascontiguousarray(root_ang_vel, dtype=np.float32)
    )
    model.forward_kinematic(data)

    body_ids = []
    for name in body_names:
        body_id = model.get_link_index(name)
        if body_id is None or body_id < 0:
            raise ValueError(f"Body {name!r} not found in Motrix model")
        body_ids.append(int(body_id))
    poses = np.asarray(model.get_link_poses(data), dtype=np.float32)[:, body_ids]
    quat_wxyz = poses[..., (6, 3, 4, 5)]
    return MotrixReferenceKinematics(
        body_pos_w=np.ascontiguousarray(poses[..., :3]),
        body_quat_w=np.ascontiguousarray(quat_wxyz),
        body_lin_vel_w=np.ascontiguousarray(
            np.asarray(model.get_link_linear_velocities(data), dtype=np.float32)[:, body_ids]
        ),
        body_ang_vel_w=np.ascontiguousarray(
            np.asarray(model.get_link_angular_velocities(data), dtype=np.float32)[:, body_ids]
        ),
    )
