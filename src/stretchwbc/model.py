"""Kinematic model of a Hello Robot Stretch 3 as a mobile manipulator.

Eight degrees of freedom, treated as one system rather than a base that drives
and then an arm that reaches:

    0  x            base position, world frame
    1  y
    2  theta        base heading
    3  lift         vertical prismatic
    4  arm          telescoping prismatic, along the base's +y (left)
    5  wrist_yaw
    6  wrist_pitch
    7  wrist_roll

**Why whole-body rather than base-then-arm.** Stretch's arm is almost entirely
prismatic and extends sideways, so its reachable set from a fixed base is close
to a plane. Planning the base first and the arm second means committing to a
base pose before knowing whether the arm can finish the job, and the usual
result is a base that parks slightly wrong and an arm that then cannot reach.
Solving for all eight at once removes the commitment.

**The consequence that makes it interesting.** The base is redundant with the
arm in one direction — sliding the base left is nearly the same end-effector
motion as extending the arm — so the Jacobian is genuinely rank-deficient in
places rather than merely ill-conditioned. Every solver in :mod:`stretchwbc.ik`
has to survive that, and it is the thing the benchmark is really measuring.

Angles are radians, lengths metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_JOINTS = 8

JOINT_NAMES = (
    "base_x",
    "base_y",
    "base_theta",
    "lift",
    "arm",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
)

#: Indices, so callers never index by a bare integer.
BASE_X, BASE_Y, BASE_THETA, LIFT, ARM, WRIST_YAW, WRIST_PITCH, WRIST_ROLL = range(N_JOINTS)


@dataclass(frozen=True)
class JointLimits:
    """Per-joint position and velocity limits.

    The base pose entries are not physical limits — a mobile base has no
    bounded x — but a bounded workspace is what keeps the solvers from
    wandering off, so they are set to the operating area and documented as
    such.
    """

    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        for name in ("lower", "upper", "velocity"):
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != (N_JOINTS,):
                raise ValueError(f"{name} must have {N_JOINTS} entries, got {array.shape}")
        if np.any(self.lower >= self.upper):
            raise ValueError("every lower limit must be below its upper limit")
        if np.any(np.asarray(self.velocity) <= 0):
            raise ValueError("velocity limits must be positive")

    @classmethod
    def stretch3(cls, workspace_m: float = 3.0) -> JointLimits:
        return cls(
            lower=np.array(
                [-workspace_m, -workspace_m, -np.pi, 0.15, 0.0, -1.75, -1.57, -3.14]
            ),
            upper=np.array([workspace_m, workspace_m, np.pi, 1.10, 0.52, 4.00, 0.56, 3.14]),
            velocity=np.array([0.25, 0.25, 0.60, 0.15, 0.15, 1.50, 1.50, 1.50]),
        )

    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=float), self.lower, self.upper)

    def violations(self, q: np.ndarray, *, tolerance: float = 0.0) -> np.ndarray:
        """Per-joint amount by which q exceeds its limits. Zero where inside."""
        q = np.asarray(q, dtype=float)
        below = np.maximum(self.lower - q - tolerance, 0.0)
        above = np.maximum(q - self.upper - tolerance, 0.0)
        return below + above

    def within(self, q: np.ndarray, *, tolerance: float = 1e-9) -> bool:
        return bool(np.all(self.violations(q, tolerance=tolerance) == 0.0))


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


@dataclass
class StretchModel:
    """Forward kinematics and Jacobian for the eight-DOF system.

    Args:
        mast_offset_xy: where the lift mast sits relative to the base origin.
        arm_retracted_y: arm carriage offset along the base's +y when fully
            retracted.
        wrist_length: wrist origin to tool tip.
        limits: joint limits; defaults to the Stretch 3 values.
    """

    mast_offset_xy: tuple[float, float] = (-0.10, 0.14)
    arm_retracted_y: float = 0.08
    wrist_length: float = 0.22
    limits: JointLimits = field(default_factory=JointLimits.stretch3)

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """End-effector pose from a configuration.

        Returns:
            Tool-tip position in the world frame, and the tool orientation as a
            3x3 rotation matrix.
        """
        q = np.asarray(q, dtype=float)
        if q.shape != (N_JOINTS,):
            raise ValueError(f"q must have {N_JOINTS} entries, got {q.shape}")

        base_rotation = _rotation_z(q[BASE_THETA])
        base_position = np.array([q[BASE_X], q[BASE_Y], 0.0])

        # Mast, then lift, then the telescoping arm extending along base +y.
        in_base = np.array(
            [
                self.mast_offset_xy[0],
                self.mast_offset_xy[1] + self.arm_retracted_y + q[ARM],
                q[LIFT],
            ]
        )

        wrist_rotation = (
            _rotation_z(q[WRIST_YAW]) @ _rotation_y(q[WRIST_PITCH]) @ _rotation_x(q[WRIST_ROLL])
        )
        tool_in_wrist = wrist_rotation @ np.array([self.wrist_length, 0.0, 0.0])

        position = base_position + base_rotation @ (in_base + tool_in_wrist)
        orientation = base_rotation @ wrist_rotation
        return position, orientation

    def position(self, q: np.ndarray) -> np.ndarray:
        return self.forward(q)[0]

    def position_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Analytic 3x8 Jacobian of tool-tip position.

        Derived rather than finite-differenced because it is called inside
        every solver iteration, and because an analytic Jacobian that disagrees
        with the numerical one is a loud, testable signal that the FK and the
        derivative have drifted apart. The test suite checks exactly that.
        """
        q = np.asarray(q, dtype=float)
        if q.shape != (N_JOINTS,):
            raise ValueError(f"q must have {N_JOINTS} entries, got {q.shape}")

        theta = q[BASE_THETA]
        base_rotation = _rotation_z(theta)
        d_base_rotation = np.array(
            [
                [-np.sin(theta), -np.cos(theta), 0.0],
                [np.cos(theta), -np.sin(theta), 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        yaw, pitch, roll = q[WRIST_YAW], q[WRIST_PITCH], q[WRIST_ROLL]
        rz, ry, rx = _rotation_z(yaw), _rotation_y(pitch), _rotation_x(roll)
        tool_axis = np.array([self.wrist_length, 0.0, 0.0])

        in_base = np.array(
            [
                self.mast_offset_xy[0],
                self.mast_offset_xy[1] + self.arm_retracted_y + q[ARM],
                q[LIFT],
            ]
        )
        tool_in_wrist = rz @ ry @ rx @ tool_axis
        local = in_base + tool_in_wrist

        jacobian = np.zeros((3, N_JOINTS))

        jacobian[:, BASE_X] = [1.0, 0.0, 0.0]
        jacobian[:, BASE_Y] = [0.0, 1.0, 0.0]
        jacobian[:, BASE_THETA] = d_base_rotation @ local
        jacobian[:, LIFT] = base_rotation @ np.array([0.0, 0.0, 1.0])
        jacobian[:, ARM] = base_rotation @ np.array([0.0, 1.0, 0.0])

        d_rz = np.array(
            [
                [-np.sin(yaw), -np.cos(yaw), 0.0],
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        d_ry = np.array(
            [
                [-np.sin(pitch), 0.0, np.cos(pitch)],
                [0.0, 0.0, 0.0],
                [-np.cos(pitch), 0.0, -np.sin(pitch)],
            ]
        )
        d_rx = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, -np.sin(roll), -np.cos(roll)],
                [0.0, np.cos(roll), -np.sin(roll)],
            ]
        )

        jacobian[:, WRIST_YAW] = base_rotation @ (d_rz @ ry @ rx @ tool_axis)
        jacobian[:, WRIST_PITCH] = base_rotation @ (rz @ d_ry @ rx @ tool_axis)
        jacobian[:, WRIST_ROLL] = base_rotation @ (rz @ ry @ d_rx @ tool_axis)

        return jacobian

    def numerical_position_jacobian(self, q: np.ndarray, *, step: float = 1e-6) -> np.ndarray:
        """Central-difference Jacobian, for testing the analytic one."""
        q = np.asarray(q, dtype=float)
        jacobian = np.zeros((3, N_JOINTS))

        for j in range(N_JOINTS):
            forward_q, backward_q = q.copy(), q.copy()
            forward_q[j] += step
            backward_q[j] -= step
            jacobian[:, j] = (self.position(forward_q) - self.position(backward_q)) / (2 * step)

        return jacobian

    def manipulability(self, q: np.ndarray) -> float:
        """Yoshikawa's measure, sqrt(det(J Jt)).

        Goes to zero at a singularity. Worth watching on this robot because the
        base-sliding-versus-arm-extending redundancy makes near-singular
        configurations common rather than exotic.
        """
        jacobian = self.position_jacobian(q)
        gram = jacobian @ jacobian.T
        return float(np.sqrt(max(np.linalg.det(gram), 0.0)))

    def random_configuration(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.limits.lower, self.limits.upper)

    def nearby_target(
        self, q: np.ndarray, rng: np.random.Generator, *, scale: float = 0.15
    ) -> np.ndarray:
        """A target reachable from ``q`` within a short horizon.

        Sampling targets uniformly over the workspace produces ones several
        metres away, and with a 0.25 m/s base and a 0.15 m/s lift a two-second
        episode cannot get there. An episode the expert itself cannot finish is
        worthless as a demonstration — the clone learns from a recording of
        failure — so targets for finite-horizon work are drawn by perturbing
        the current configuration instead.

        ``scale`` is the fraction of each joint's range the perturbation may
        span.
        """
        q = np.asarray(q, dtype=float)
        span = (self.limits.upper - self.limits.lower) * scale
        perturbed = self.limits.clamp(q + rng.uniform(-span, span))
        return self.position(perturbed)

    def random_reachable_target(self, rng: np.random.Generator) -> np.ndarray:
        """A target position generated from a valid configuration.

        Sampling targets in Cartesian space would produce unreachable ones, and
        a solver benchmark that includes unreachable targets measures how each
        solver fails rather than how well it succeeds. Both are worth knowing,
        but not mixed together in one number.
        """
        return self.position(self.random_configuration(rng))
