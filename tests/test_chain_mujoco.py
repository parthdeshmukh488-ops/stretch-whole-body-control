"""The exact chain against MuJoCo, and its Jacobian against finite differences.

The MuJoCo comparison needs the `mujoco` package and the Menagerie model; it is
skipped without them. Set STRETCH3_XML to the model path, or put Menagerie at
third_party/mujoco_menagerie (tools/fetch_menagerie.py does that).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from stretchwbc.chain import Stretch3Chain, differential_drive_map

ROOT = Path(__file__).resolve().parents[1]
XML = Path(
    os.environ.get(
        "STRETCH3_XML",
        ROOT / "third_party" / "mujoco_menagerie" / "hello_robot_stretch_3" / "stretch.xml",
    )
)


def mujoco_tool_pose(mujoco, model, data, q):
    """Set the Menagerie model to configuration q and return the grasp centre pose."""
    data.qpos[:] = 0.0
    data.qpos[0:3] = [q[0], q[1], 0.0]
    data.qpos[3:7] = [np.cos(q[2] / 2), 0.0, 0.0, np.sin(q[2] / 2)]  # yaw about z (w, x, y, z)
    data.qpos[model.joint("joint_lift").qposadr[0]] = q[3]
    for segment in ("l3", "l2", "l1", "l0"):
        data.qpos[model.joint(f"joint_arm_{segment}").qposadr[0]] = q[4] / 4
    data.qpos[model.joint("joint_wrist_yaw").qposadr[0]] = q[5]
    data.qpos[model.joint("joint_wrist_pitch").qposadr[0]] = q[6]
    data.qpos[model.joint("joint_wrist_roll").qposadr[0]] = q[7]
    mujoco.mj_kinematics(model, data)
    body = model.body("link_grasp_center").id
    return data.xpos[body].copy(), data.xmat[body].reshape(3, 3).copy()


def test_forward_kinematics_match_mujoco():
    mujoco = pytest.importorskip("mujoco")
    if not XML.exists():
        pytest.skip(f"Menagerie Stretch 3 model not found at {XML}")
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    chain = Stretch3Chain()
    rng = np.random.default_rng(0)
    worst_position, worst_rotation = 0.0, 0.0
    for _ in range(500):
        q = chain.random_configuration(rng)
        expected_p, expected_r = mujoco_tool_pose(mujoco, model, data, q)
        p, r = chain.forward(q)
        worst_position = max(worst_position, float(np.linalg.norm(p - expected_p)))
        worst_rotation = max(worst_rotation, float(np.abs(r - expected_r).max()))
    assert worst_position < 1e-6  # metres: a micrometre, far under the millimetre claimed
    assert worst_rotation < 1e-6


def test_geometric_jacobian_matches_finite_differences():
    chain = Stretch3Chain()
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = chain.random_configuration(rng)
        assert np.allclose(
            chain.position_jacobian(q), chain.numerical_position_jacobian(q), atol=1e-6
        )


def test_arm_extends_to_the_right():
    chain = Stretch3Chain()
    q = np.zeros(8)
    q[3] = 0.5
    retracted = chain.position(q)
    q[4] = 0.4
    extended = chain.position(q)
    assert np.allclose(extended - retracted, [0.0, -0.4, 0.0], atol=1e-9)


def test_differential_drive_cannot_slide_sideways():
    chain = Stretch3Chain()
    q = np.zeros(8)
    q[3], q[4] = 0.6, 0.2
    j = chain.position_jacobian(q) @ differential_drive_map(q[2])
    assert j.shape == (3, 7)
    # Base velocity v at heading 0 moves the tool along x only; no column is pure +y from the base.
    assert np.allclose(j[:, 0], [1.0, 0.0, 0.0])
