"""Cold-path MuJoCo forward kinematics for reference-motion conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unisim.backend.base import SimBackend
from unisim.utils.rotation import np_quat_apply_inverse_batched


@dataclass(frozen=True)
class MujocoReferenceKinematics:
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray


def materialize_reference_kinematics(
    backend: SimBackend,
    *,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    root_lin_vel: np.ndarray,
    root_ang_vel: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    body_names: tuple[str, ...],
) -> MujocoReferenceKinematics:
    """Run ``mj_forward`` once per source frame without stepping dynamics.

    ``root_ang_vel`` is the world-frame root angular velocity. MuJoCo's
    free-joint ``qvel[3:6]`` slot is body-local, so it is rotated into the
    root frame here; the returned body velocities are world-frame throughout.
    """
    if backend.backend_type != "mujoco":
        raise ValueError("MuJoCo reference materialization requires the MuJoCo backend")

    import mujoco

    model = backend._model  # pyright: ignore[reportPrivateUsage]
    count = int(root_pos.shape[0])
    expected = {
        "root_pos": (count, 3),
        "root_quat_wxyz": (count, 4),
        "root_lin_vel": (count, 3),
        "root_ang_vel": (count, 3),
        "joint_pos": (count, model.nu),
        "joint_vel": (count, model.nu),
    }
    values = {
        "root_pos": root_pos,
        "root_quat_wxyz": root_quat_wxyz,
        "root_lin_vel": root_lin_vel,
        "root_ang_vel": root_ang_vel,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
    }
    for name, shape in expected.items():
        value = np.asarray(values[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite with shape {shape}, got {value.shape}")

    body_ids = np.asarray(backend.get_body_ids(body_names), dtype=np.intp)
    actuator_joint_ids = model.actuator_trnid[:, 0].astype(np.intp)
    qpos_ids = model.jnt_qposadr[actuator_joint_ids]
    qvel_ids = model.jnt_dofadr[actuator_joint_ids]
    quat = np.asarray(root_quat_wxyz, dtype=np.float64)
    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
    root_ang_vel_local = np_quat_apply_inverse_batched(
        quat, np.asarray(root_ang_vel, dtype=np.float64)
    ).astype(np.float32)
    pos = np.empty((count, len(body_ids), 3), dtype=np.float32)
    quat = np.empty((count, len(body_ids), 4), dtype=np.float32)
    lin_vel = np.empty_like(pos)
    ang_vel = np.empty_like(pos)
    data = mujoco.MjData(model)
    for frame in range(count):
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat_wxyz[frame]
        data.qvel[:3] = root_lin_vel[frame]
        data.qvel[3:6] = root_ang_vel_local[frame]
        data.qpos[qpos_ids] = joint_pos[frame]
        data.qvel[qvel_ids] = joint_vel[frame]
        mujoco.mj_forward(model, data)
        pos[frame] = data.xpos[body_ids]
        quat[frame] = data.xquat[body_ids]
        for column, body_id in enumerate(body_ids):
            velocity = np.empty(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), velocity, 0
            )
            ang_vel[frame, column] = velocity[:3]
            lin_vel[frame, column] = velocity[3:]
    return MujocoReferenceKinematics(pos, quat, lin_vel, ang_vel)


__all__ = ["MujocoReferenceKinematics", "materialize_reference_kinematics"]
