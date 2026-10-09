"""Regression tests for reference-motion velocity frame handling.

MuJoCo's free-joint ``qvel[3:6]`` slot is body-local while the materializer
contract takes world-frame ``root_ang_vel``.  These tests pin the world-frame
round trip through the engine: the naive slot write returned ``R @ omega``
(rotated by the root orientation) instead of ``omega``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
import mujoco

from unisim.backend.mujoco.reference_motion import materialize_reference_kinematics

_XML = """
<mujoco>
  <worldbody>
    <body name="pelvis" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.1"/>
      <body name="child" pos="0.2 0 0">
        <geom type="sphere" size="0.05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class _StubMujocoBackend:
    backend_type = "mujoco"

    def __init__(self, model: mujoco.MjModel):
        self._model = model

    def get_body_ids(self, names) -> np.ndarray:
        return np.array(
            [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names],
            dtype=np.int32,
        )


def test_materialize_keeps_world_frame_angular_velocity() -> None:
    model = mujoco.MjModel.from_xml_string(_XML)
    backend = _StubMujocoBackend(model)
    count = 3
    lin_vel = np.array([0.3, 0.0, 0.0])
    half = np.sqrt(0.5)
    # (root quat, world omega, expected lever-arm velocity of the child at
    # body-frame offset (0.2, 0, 0)): omega x (R @ offset).
    cases = [
        (np.array([half, 0.0, 0.0, half]), np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, 0.1])),
        (np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.1, 0.0])),
        (np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.5]), np.array([0.0, -0.1, 0.0])),
    ]
    for quat, omega_world, child_lever in cases:
        kinematics = materialize_reference_kinematics(
            backend,
            root_pos=np.tile(np.array([0.0, 0.0, 1.0]), (count, 1)),
            root_quat_wxyz=np.tile(quat, (count, 1)),
            root_lin_vel=np.tile(lin_vel, (count, 1)),
            root_ang_vel=np.tile(omega_world, (count, 1)),
            joint_pos=np.zeros((count, model.nu)),
            joint_vel=np.zeros((count, model.nu)),
            body_names=("pelvis", "child"),
        )
        np.testing.assert_allclose(
            kinematics.body_ang_vel_w[:, 0], np.tile(omega_world, (count, 1)), atol=1e-6
        )
        np.testing.assert_allclose(
            kinematics.body_lin_vel_w[:, 0], np.tile(lin_vel, (count, 1)), atol=1e-6
        )
        np.testing.assert_allclose(
            kinematics.body_lin_vel_w[:, 1],
            np.tile(lin_vel + child_lever, (count, 1)),
            atol=1e-5,
        )
