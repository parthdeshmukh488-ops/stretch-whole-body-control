"""Stretch 3 forward kinematics from the real geometry, as a chain of 4x4 transforms.

:class:`stretchwbc.model.StretchModel` is a simplified analytic stand-in with
plausible dimensions. This class uses the geometry of MuJoCo Menagerie's
``hello_robot_stretch_3`` model instead (commit f054586a, extracted with
``tools/extract_stretch3_chain.py``), so its forward kinematics agree with
MuJoCo's: ``tests/test_chain_mujoco.py`` checks that to a micrometre.

It has the same interface as StretchModel (``position``, ``position_jacobian``,
``limits``, ``random_configuration``, ...), so the solvers in :mod:`stretchwbc.ik`
run on either. The configuration is the same eight values:

    x, y, theta   base pose in the plane
    lift          0 .. 1.1 m
    arm           0 .. 0.52 m, shared equally by the four telescoping segments
    wrist yaw, pitch, roll

Two differences from the simplified model, both from the real robot: the arm
extends to the robot's **right** (-y), and the gripper hangs below the wrist,
pointing along the arm when the wrist yaw is zero.

**Holonomic base, as in StretchModel.** The base is treated as able to move in
x and y independently. The real Stretch has a differential drive and cannot
move sideways; a planner for the real robot needs that constraint (see
:func:`stretchwbc.chain.differential_drive_map`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stretchwbc.model import N_JOINTS, JointLimits

# Body offsets from the parent (position, quaternion w x y z) and joints (type, axis),
# from base_link to link_grasp_center. "arm" joints share the arm value equally.
CHAIN = (
    ("link_lift", [-0.104385, 0.134999, 0.2], [0.7071067812, 0.0, 0.0, 0.7071067812], ("lift", "slide", [0.0, 0.0, 1.0])),
    ("link_arm_l4", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], None),
    ("link_arm_l3", [-0.2677, 0.0, 0.0], [0.5, 0.5, -0.5, -0.5], ("arm", "slide", [0.0, 0.0, 1.0])),
    ("link_arm_l2", [0.0, 0.0, 0.013], [1.0, 0.0, 0.0, 0.0], ("arm", "slide", [0.0, 0.0, 1.0])),
    ("link_arm_l1", [0.0, 0.0, 0.013], [1.0, 0.0, 0.0, 0.0], ("arm", "slide", [0.0, 0.0, 1.0])),
    ("link_arm_l0", [0.0, 0.0, -0.01375], [1.0, 0.0, 0.0, 0.0], ("arm", "slide", [0.0, 0.0, 1.0])),
    ("link_wrist_yaw", [0.083, -0.03075, 0.0], [-0.7071067812, -0.7071067812, 0.0, 0.0], ("wrist_yaw", "hinge", [0.0, 0.0, -1.0])),
    ("link_DW3_wrist_pitch", [-0.01946, 0.0, 0.0305], [0.5003981634, -0.4996018366, 0.4999998415, -0.4999998415], ("wrist_pitch", "hinge", [0.0, 0.0, -1.0])),
    ("link_SG3_gripper_body", [-0.0399, -0.024, 0.0196], [0.7071067812, 0.0, -0.7071067812, 0.0], ("wrist_roll", "hinge", [0.0, 0.0, 1.0])),
    ("link_grasp_center", [0.0, 0.0, 0.23], [0.5, -0.5, -0.5, 0.5], None),
)  # fmt: skip

JOINT_INDEX = {"lift": 3, "arm": 4, "wrist_yaw": 5, "wrist_pitch": 6, "wrist_roll": 7}
ARM_SEGMENTS = 4


def _quat_to_rotation(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def _transform(rotation: np.ndarray, translation) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


_FIXED = [_transform(_quat_to_rotation(quat), pos) for _, pos, quat, _ in CHAIN]


def stretch3_limits(workspace_m: float = 3.0) -> JointLimits:
    """Joint ranges from the Menagerie model; base and velocity limits as in StretchModel."""
    base = JointLimits.stretch3(workspace_m)
    lower = base.lower.copy()
    upper = base.upper.copy()
    lower[3:], upper[3:] = [0.0, 0.0, -1.39, -1.57, -3.14], [1.1, 0.52, 4.42, 0.56, 3.14]
    return JointLimits(lower=lower, upper=upper, velocity=base.velocity.copy())


@dataclass
class Stretch3Chain:
    limits: JointLimits = field(default_factory=stretch3_limits)

    def frames(self, q: np.ndarray) -> list[tuple[np.ndarray, tuple | None]]:
        """World transform of every chain body after its joint, and that body's joint."""
        q = np.asarray(q, dtype=float)
        if q.shape != (N_JOINTS,):
            raise ValueError(f"q must have {N_JOINTS} entries, got {q.shape}")
        current = _transform(_axis_rotation(np.array([0.0, 0.0, 1.0]), q[2]), [q[0], q[1], 0.0])
        out = []
        for fixed, (_, _, _, joint) in zip(_FIXED, CHAIN, strict=True):
            current = current @ fixed
            if joint is not None:
                name, kind, axis = joint
                value = q[JOINT_INDEX[name]] / (ARM_SEGMENTS if name == "arm" else 1)
                axis = np.asarray(axis, dtype=float)
                if kind == "slide":
                    motion = _transform(np.eye(3), axis * value)
                else:
                    motion = _transform(_axis_rotation(axis, value), np.zeros(3))
                current = current @ motion
            out.append((current, joint))
        return out

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tool = self.frames(q)[-1][0]
        return tool[:3, 3].copy(), tool[:3, :3].copy()

    def position(self, q: np.ndarray) -> np.ndarray:
        return self.forward(q)[0]

    def position_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Geometric 3x8 Jacobian: a slide joint moves the tool along its axis; a hinge
        moves it by axis x (tool - joint origin). The four arm segments each contribute
        a quarter, because they share one arm value."""
        q = np.asarray(q, dtype=float)
        frames = self.frames(q)
        tool = frames[-1][0][:3, 3]
        jacobian = np.zeros((3, N_JOINTS))
        jacobian[:, 0] = [1.0, 0.0, 0.0]
        jacobian[:, 1] = [0.0, 1.0, 0.0]
        jacobian[:, 2] = np.cross([0.0, 0.0, 1.0], tool - np.array([q[0], q[1], 0.0]))
        for frame, joint in frames:
            if joint is None:
                continue
            name, kind, axis = joint
            world_axis = frame[:3, :3] @ np.asarray(axis, dtype=float)
            column = (
                world_axis if kind == "slide" else np.cross(world_axis, tool - frame[:3, 3])
            )
            jacobian[:, JOINT_INDEX[name]] += column / (ARM_SEGMENTS if name == "arm" else 1)
        return jacobian

    def numerical_position_jacobian(self, q: np.ndarray, *, step: float = 1e-6) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        jacobian = np.zeros((3, N_JOINTS))
        for j in range(N_JOINTS):
            plus, minus = q.copy(), q.copy()
            plus[j] += step
            minus[j] -= step
            jacobian[:, j] = (self.position(plus) - self.position(minus)) / (2 * step)
        return jacobian

    def random_configuration(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.limits.lower, self.limits.upper)

    def random_reachable_target(self, rng: np.random.Generator) -> np.ndarray:
        return self.position(self.random_configuration(rng))


def differential_drive_map(theta: float) -> np.ndarray:
    """8x7 map from (v, omega, lift, arm, yaw, pitch, roll) to the joint velocities.

    A differential-drive base moves only along its heading (v) and turns in place
    (omega): x_dot = v cos(theta), y_dot = v sin(theta). Multiplying a Jacobian by this
    map gives the Jacobian a planner for the real Stretch should use: one fewer column,
    and no sideways motion.
    """
    s = np.zeros((N_JOINTS, N_JOINTS - 1))
    s[0, 0], s[1, 0] = np.cos(theta), np.sin(theta)
    s[2, 1] = 1.0
    s[3:, 2:] = np.eye(5)
    return s
