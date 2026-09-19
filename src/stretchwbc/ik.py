"""Three inverse-kinematics solvers, built to be compared.

All three take the same signature and return the same result type, so the
benchmark in ``examples/`` can swap them without special-casing. They differ in
exactly one place — how the joint step is computed from the position error —
which is what makes the comparison mean something.

- **Jacobian transpose.** `dq = alpha * J.T @ e`. No matrix inverse, so it
  cannot blow up at a singularity. It is also slow to converge and the step
  direction is only loosely related to the error direction.
- **Damped least squares.** `dq = J.T (J J.T + lambda^2 I)^-1 e`. The standard
  answer. The damping term trades accuracy for stability near singularities,
  and with a fixed damping you pay that trade everywhere, including in the
  well-conditioned interior of the workspace where you did not need it.
- **Selectively damped least squares.** Damping applied per singular value
  rather than uniformly, so well-conditioned directions are solved crisply and
  only the near-singular ones are damped. More arithmetic per iteration; the
  question the benchmark answers is whether it buys enough to be worth it on
  this robot.

Joint limits are enforced by clamping after each step rather than inside it.
That is the honest simple choice and it has a known failure mode: a joint
pinned at its limit keeps being pushed into it and the solver stalls. The
:mod:`stretchwbc.safety` filter is what handles limits properly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import N_JOINTS, StretchModel


@dataclass(frozen=True)
class IKResult:
    """Outcome of one IK solve."""

    q: np.ndarray
    success: bool
    error_m: float
    iterations: int
    method: str

    def __str__(self) -> str:
        state = "converged" if self.success else "failed"
        return (
            f"{self.method}: {state} in {self.iterations} iters, {self.error_m * 1000:.2f} mm"
        )


def _clamped_step(
    model: StretchModel, q: np.ndarray, dq: np.ndarray, max_step: float
) -> np.ndarray:
    """Apply a step, scaled to a maximum norm, then clamp to joint limits."""
    norm = float(np.linalg.norm(dq))
    if norm > max_step and norm > 0.0:
        dq = dq * (max_step / norm)
    return model.limits.clamp(q + dq)


def solve_jacobian_transpose(
    model: StretchModel,
    target: np.ndarray,
    q0: np.ndarray,
    *,
    tolerance: float = 1e-3,
    max_iterations: int = 500,
    max_step: float = 0.05,
) -> IKResult:
    """Jacobian-transpose IK.

    The step size is chosen per iteration by the standard closed form that
    minimises the residual along the transpose direction, rather than a fixed
    gain. A fixed gain is either unstable far from the target or glacial near
    it, and tuning one per robot is exactly the kind of hidden constant that
    makes a benchmark meaningless.
    """
    q = model.limits.clamp(np.asarray(q0, dtype=float).copy())
    target = np.asarray(target, dtype=float)

    for iteration in range(1, max_iterations + 1):
        error = target - model.position(q)
        error_norm = float(np.linalg.norm(error))

        if error_norm < tolerance:
            return IKResult(q, True, error_norm, iteration, "jacobian_transpose")

        jacobian = model.position_jacobian(q)
        direction = jacobian.T @ error
        projected = jacobian @ direction

        denominator = float(projected @ projected)
        if denominator < 1e-18:
            break

        alpha = float(error @ projected) / denominator
        q = _clamped_step(model, q, alpha * direction, max_step)

    error_norm = float(np.linalg.norm(target - model.position(q)))
    return IKResult(q, error_norm < tolerance, error_norm, max_iterations, "jacobian_transpose")


def solve_damped_least_squares(
    model: StretchModel,
    target: np.ndarray,
    q0: np.ndarray,
    *,
    tolerance: float = 1e-3,
    max_iterations: int = 200,
    damping: float = 0.05,
    max_step: float = 0.05,
) -> IKResult:
    """Damped least squares, with a fixed damping factor."""
    q = model.limits.clamp(np.asarray(q0, dtype=float).copy())
    target = np.asarray(target, dtype=float)
    identity = np.eye(3)

    for iteration in range(1, max_iterations + 1):
        error = target - model.position(q)
        error_norm = float(np.linalg.norm(error))

        if error_norm < tolerance:
            return IKResult(q, True, error_norm, iteration, "damped_least_squares")

        jacobian = model.position_jacobian(q)
        gram = jacobian @ jacobian.T + (damping**2) * identity

        try:
            dq = jacobian.T @ np.linalg.solve(gram, error)
        except np.linalg.LinAlgError:
            break

        q = _clamped_step(model, q, dq, max_step)

    error_norm = float(np.linalg.norm(target - model.position(q)))
    return IKResult(
        q, error_norm < tolerance, error_norm, max_iterations, "damped_least_squares"
    )


def solve_selectively_damped(
    model: StretchModel,
    target: np.ndarray,
    q0: np.ndarray,
    *,
    tolerance: float = 1e-3,
    max_iterations: int = 200,
    singular_floor: float = 1e-2,
    max_step: float = 0.05,
) -> IKResult:
    """Least squares with damping applied per singular value.

    Each singular direction gets `s / (s^2 + eps^2)` where eps is zero for
    directions comfortably above the floor and ramps in only as a direction
    approaches degeneracy. Well-conditioned directions are therefore solved as
    crisply as a plain pseudo-inverse would, which is the whole point.
    """
    q = model.limits.clamp(np.asarray(q0, dtype=float).copy())
    target = np.asarray(target, dtype=float)

    for iteration in range(1, max_iterations + 1):
        error = target - model.position(q)
        error_norm = float(np.linalg.norm(error))

        if error_norm < tolerance:
            return IKResult(q, True, error_norm, iteration, "selectively_damped")

        jacobian = model.position_jacobian(q)
        u, s, vt = np.linalg.svd(jacobian, full_matrices=False)

        # Ramp the damping in only where a singular value is small. Squaring
        # the ratio makes the transition smooth rather than a switch, which
        # matters because a switch produces a discontinuous joint velocity.
        ratio = np.clip(s / singular_floor, 0.0, 1.0)
        epsilon = singular_floor * (1.0 - ratio) ** 2
        inverse = s / (s**2 + epsilon**2)

        dq = vt.T @ (inverse * (u.T @ error))
        q = _clamped_step(model, q, dq, max_step)

    error_norm = float(np.linalg.norm(target - model.position(q)))
    return IKResult(q, error_norm < tolerance, error_norm, max_iterations, "selectively_damped")


SOLVERS = {
    "jacobian_transpose": solve_jacobian_transpose,
    "damped_least_squares": solve_damped_least_squares,
    "selectively_damped": solve_selectively_damped,
}


def solve(
    method: str,
    model: StretchModel,
    target: np.ndarray,
    q0: np.ndarray | None = None,
    **kwargs,
) -> IKResult:
    """Dispatch to a named solver.

    A ``q0`` of None starts from the middle of every joint range, which is a
    deliberately neutral seed: seeding from a configuration near the answer
    flatters every solver equally but hides how they behave when they have to
    travel.
    """
    if method not in SOLVERS:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(SOLVERS)}")

    if q0 is None:
        q0 = (model.limits.lower + model.limits.upper) / 2.0

    q0 = np.asarray(q0, dtype=float)
    if q0.shape != (N_JOINTS,):
        raise ValueError(f"q0 must have {N_JOINTS} entries, got {q0.shape}")

    return SOLVERS[method](model, target, q0, **kwargs)
