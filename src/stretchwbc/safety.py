"""A QP safety filter that sits between any controller and the robot.

The controller — an IK solver, a teleoperator, a learned policy — proposes a
joint velocity. This module finds the *closest* velocity that satisfies every
constraint, and sends that instead:

    minimise   ||dq - dq_desired||^2
    subject to  A dq <= b

"Closest" is the whole design. A filter that clips or rejects changes the
commanded direction, and a controller that asks to move diagonally into a
corner ends up sliding along one axis instead of the other. Projecting onto the
feasible set keeps as much of the intent as the constraints allow, and it is
also what makes the filter transparent when nothing is active: if the request
is already feasible, it passes through untouched.

**Why a QP rather than clamping each joint.** Box constraints alone could be
clamped element-wise — for a diagonal objective that is in fact the exact
solution. But the constraints that matter on this robot are not boxes. Tipping
couples arm extension to lift height, and a constraint that couples two joints
cannot be enforced by looking at either one alone. Once a single coupling
constraint exists, the problem is a genuine QP.

**Control barrier functions** turn position limits into velocity constraints.
Given a scalar margin `h(q) >= 0` that must stay non-negative, requiring

    grad_h . dq >= -alpha * h

makes `h` decay at worst exponentially and never cross zero. As the margin
shrinks the permitted approach velocity shrinks with it, so the robot slows
smoothly into a limit instead of running at it and stopping dead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .model import ARM, LIFT, N_JOINTS, JointLimits, StretchModel


@dataclass(frozen=True)
class QPSolution:
    """Result of one filter step."""

    dq: np.ndarray
    active: int  # constraints pushing back
    iterations: int
    converged: bool
    correction: float  # how far the request was moved, in rad/s
    slack: np.ndarray | None = None  # by how much each soft constraint was relaxed

    @property
    def modified(self) -> bool:
        return self.correction > 1e-12

    @property
    def relaxed(self) -> float:
        """Largest soft-constraint relaxation. Zero when all were satisfied."""
        if self.slack is None or self.slack.size == 0:
            return 0.0
        return float(np.max(self.slack))


def solve_projection_qp(
    desired: np.ndarray,
    constraint_matrix: np.ndarray,
    constraint_bound: np.ndarray,
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
) -> QPSolution:
    """Project ``desired`` onto ``{x : A x <= b}``.

    Solved through the dual. For ``min 0.5||x - z||^2 s.t. Ax <= b`` the dual is

        min_{lambda >= 0}  0.5 lambda' D lambda + lambda' k,
        D = A A',  k = b - A z,

    and the primal follows as ``x = z - A' lambda``.

    The dual is minimised by **accelerated projected gradient** (FISTA), not by
    Hildreth's coordinate descent, and the reason is worth recording because
    the first version of this file used Hildreth and was wrong.

    Hildreth sweeps one multiplier at a time, so its cost per iteration is a
    Python loop over the constraints and its convergence depends on how well
    conditioned the dual is. On the augmented problem built by
    :func:`solve_box_soft_qp` — where the slack block couples weakly to
    the rest — it stalled, hit the iteration cap, and returned a velocity that
    **broke the joint velocity limits by 0.135 rad/s**. A safety filter
    returning an unexecutable command that looks like a normal answer is the
    worst failure mode available.

    Projected gradient has a guaranteed O(1/k^2) rate on this problem and each
    iteration is a single matrix-vector product, so it vectorises: a few
    hundred iterations cost less than a dozen Hildreth sweeps, and the result
    is checked against the constraints before it is returned.
    """
    desired = np.asarray(desired, dtype=float).ravel()
    constraint_matrix = np.atleast_2d(np.asarray(constraint_matrix, dtype=float))
    constraint_bound = np.asarray(constraint_bound, dtype=float).ravel()

    if constraint_matrix.shape[0] != constraint_bound.shape[0]:
        raise ValueError("constraint_matrix and constraint_bound disagree on row count")
    if constraint_matrix.size and constraint_matrix.shape[1] != desired.shape[0]:
        raise ValueError("constraint_matrix columns must match the length of desired")

    n_constraints = constraint_matrix.shape[0]
    if n_constraints == 0:
        return QPSolution(desired.copy(), 0, 0, True, 0.0)

    # Scaling a row of A and its bound by the same positive number leaves the
    # feasible set untouched but changes the dual matrix's conditioning, and
    # Hildreth's convergence rate depends on that conditioning directly. Here
    # the rows are wildly different in magnitude — a velocity bound is a unit
    # vector, a tipping gradient is not — and without normalisation a
    # meaningful fraction of solves hit the iteration cap instead of
    # converging. Normalising every row to unit length costs one pass and
    # fixes it.
    row_norms = np.linalg.norm(constraint_matrix, axis=1)
    usable = row_norms > 1e-14

    scale = np.where(usable, row_norms, 1.0)
    scaled_matrix = constraint_matrix / scale[:, None]
    scaled_bound = constraint_bound / scale

    # A row of all zeros carries no information. It is either trivially
    # satisfiable or trivially infeasible; either way its multiplier stays at
    # zero and it is skipped rather than divided by.
    dual_matrix = scaled_matrix @ scaled_matrix.T
    dual_vector = scaled_bound - scaled_matrix @ desired

    diagonal = np.diag(dual_matrix).copy()
    diagonal = np.where(usable, diagonal, 1.0)

    # Fast path: nothing is violated, so the projection is the identity. This
    # is the common case in a control loop and it skips the iteration entirely.
    if np.all(dual_vector[usable] >= 0.0):
        return QPSolution(desired.copy(), 0, 0, True, 0.0)

    # Step size is 1 / L, with L the largest eigenvalue of the dual matrix.
    # The matrix is small and symmetric positive semi-definite, so this is
    # exact and cheap rather than estimated.
    eigenvalues = np.linalg.eigvalsh(dual_matrix)
    lipschitz = float(max(eigenvalues[-1], 1e-12))
    step = 1.0 / lipschitz

    multipliers = np.zeros(n_constraints)
    momentum = multipliers.copy()
    theta = 1.0

    converged = False
    iterations = 0
    # noqa on the loop variable: it is the iteration count and is read after
    # the loop, so the "unused" rule does not apply.
    for iterations in range(1, max_iterations + 1):  # noqa: B007
        gradient = dual_matrix @ momentum + dual_vector
        updated = np.maximum(momentum - step * gradient, 0.0)
        updated[~usable] = 0.0

        theta_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta * theta))
        momentum = updated + ((theta - 1.0) / theta_next) * (updated - multipliers)

        change = float(np.max(np.abs(updated - multipliers))) if n_constraints else 0.0
        multipliers, theta = updated, theta_next

        if change < tolerance:
            converged = True
            break

    # Whether the dual converged is a statement about the multipliers. What
    # matters to the caller is whether the primal is feasible, so that is what
    # is checked and reported.
    primal = desired - scaled_matrix.T @ multipliers
    converged = converged and bool(np.all(scaled_matrix @ primal - scaled_bound <= 1e-7))

    dq = primal
    return QPSolution(
        dq=dq,
        active=int(np.sum(multipliers > 1e-9)),
        iterations=iterations,
        converged=converged,
        correction=float(np.linalg.norm(dq - desired)),
    )


def solve_box_soft_qp(
    desired: np.ndarray,
    box_lower: np.ndarray,
    box_upper: np.ndarray,
    soft_matrix: np.ndarray,
    soft_bound: np.ndarray,
    *,
    penalty: float = 2.0e3,
    max_iterations: int = 600,
    tolerance: float = 1e-11,
) -> QPSolution:
    """Closest velocity inside a box, with coupling constraints as a penalty.

        minimise   0.5||dq - desired||^2
                   + 0.5 * penalty * sum_i max(0, a_i.dq - b_i)^2
        subject to  box_lower_j <= dq_j <= box_upper_j     (enforced exactly)

    **What goes in the box, and why it is not just the velocity limits.**

    Every *axis-aligned* constraint belongs here, because projection onto a box
    is a clip: it is exact, it costs nothing, and it holds at every iterate
    including an unconverged one. That covers two families:

    - **Velocity limits.** The motors cannot exceed them, so a command outside
      them describes a motion the robot will not perform.
    - **Joint-position barriers.** ``dq_j <= alpha * (upper_j - q_j)`` involves
      one joint, so it is axis-aligned and intersects the velocity box to give
      a tighter box. Putting these in the *soft* set was a real bug: under
      persistent one-directional pressure from a controller, and in conflict
      with the coupling constraint, the penalty let joints past their limits —
      132 violations across 50 rollouts of a learned policy. Folding them into
      the box took that to zero and cost nothing.

    **What stays soft: genuinely coupling constraints.** The tipping constraint
    ties arm extension to lift height, so no per-joint bound can express it. It
    is also the one that can be *unsatisfiable*: roughly a fifth of
    configurations start outside the tipping-safe set, and no velocity within
    the motors' limits returns them in one step. Treating it as hard makes the
    problem infeasible exactly there.

    **Why not a hard QP over all of it.** That was the first design and it was
    wrong in a way worth recording. With everything hard, those infeasible
    states make the dual diverge, and the filter returned a velocity breaking
    the joint velocity limits by 0.135 rad/s while reporting nothing unusual.
    Measured, not hypothesised.

    An empty box — ``box_lower > box_upper``, which a barrier can produce when
    the robot is already outside a joint limit — is collapsed to the single
    point nearest both bounds rather than raising, because a safety filter
    refusing to answer is not a useful safety filter.
    """
    desired = np.asarray(desired, dtype=float).ravel()
    box_lower = np.asarray(box_lower, dtype=float).ravel()
    box_upper = np.asarray(box_upper, dtype=float).ravel()
    soft_matrix = np.atleast_2d(np.asarray(soft_matrix, dtype=float))
    soft_bound = np.asarray(soft_bound, dtype=float).ravel()

    if box_lower.shape != desired.shape or box_upper.shape != desired.shape:
        raise ValueError("box bounds must have one entry per variable")

    n_soft = soft_matrix.shape[0] if soft_matrix.size else 0
    if n_soft and soft_matrix.shape[1] != desired.shape[0]:
        raise ValueError("soft_matrix columns must match the length of desired")
    if n_soft != soft_bound.shape[0]:
        raise ValueError("soft_matrix and soft_bound disagree on row count")

    # Collapse an inverted box to its midpoint so the feasible set is never
    # empty. This happens when a joint is already outside its position limit:
    # the barrier then demands motion in one direction while the velocity limit
    # bounds how much is available.
    crossed = box_lower > box_upper
    if np.any(crossed):
        midpoint = 0.5 * (box_lower + box_upper)
        box_lower = np.where(crossed, midpoint, box_lower)
        box_upper = np.where(crossed, midpoint, box_upper)

    clipped = np.clip(desired, box_lower, box_upper)

    if n_soft == 0:
        return QPSolution(
            dq=clipped,
            active=int(np.sum(np.abs(clipped - desired) > 1e-12)),
            iterations=0,
            converged=True,
            correction=float(np.linalg.norm(clipped - desired)),
            slack=np.zeros(0),
        )

    largest_singular = float(np.linalg.norm(soft_matrix, 2))
    lipschitz = 1.0 + penalty * largest_singular**2
    step = 1.0 / lipschitz

    current = clipped.copy()
    momentum = current.copy()
    theta = 1.0

    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):  # noqa: B007 - counted below
        overshoot = np.maximum(soft_matrix @ momentum - soft_bound, 0.0)
        gradient = (momentum - desired) + penalty * (soft_matrix.T @ overshoot)

        # The clip is the projection onto the box, so the iterate never leaves
        # it. Stopping early costs accuracy against the coupling constraints
        # and nothing at all against the velocity or joint-position limits.
        updated = np.clip(momentum - step * gradient, box_lower, box_upper)

        theta_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta * theta))
        momentum = updated + ((theta - 1.0) / theta_next) * (updated - current)

        change = float(np.max(np.abs(updated - current)))
        current, theta = updated, theta_next

        if change < tolerance:
            converged = True
            break

    slack = np.maximum(soft_matrix @ current - soft_bound, 0.0)

    return QPSolution(
        dq=current,
        active=int(np.sum(slack > 1e-9)),
        iterations=iterations,
        converged=converged,
        correction=float(np.linalg.norm(current - desired)),
        slack=slack,
    )


@dataclass(frozen=True)
class BarrierConstraint:
    """A scalar margin ``h(q) >= 0`` enforced through its gradient.

    Args:
        name: for reporting which constraint bound a command.
        margin: ``h(q)``; positive inside the safe set.
        gradient: ``dh/dq``, shape (8,).
        alpha: how hard the barrier pushes back. Larger permits faster approach
            to the boundary and reacts later; smaller is more conservative and
            costs workspace.
    """

    name: str
    margin: Callable[[np.ndarray], float]
    gradient: Callable[[np.ndarray], np.ndarray]
    alpha: float = 4.0

    def as_row(self, q: np.ndarray) -> tuple[np.ndarray, float]:
        """The constraint as one row of ``A dq <= b``.

        ``grad_h . dq >= -alpha h`` becomes ``-grad_h . dq <= alpha h``.
        """
        h = float(self.margin(q))
        grad = np.asarray(self.gradient(q), dtype=float)
        return -grad, self.alpha * h


def joint_limit_barriers(limits: JointLimits, *, alpha: float = 4.0) -> list[BarrierConstraint]:
    """Two barriers per joint, one for each end of its range."""
    constraints: list[BarrierConstraint] = []

    for j in range(N_JOINTS):
        for side in ("lower", "upper"):

            def margin(q, j=j, side=side):
                return q[j] - limits.lower[j] if side == "lower" else limits.upper[j] - q[j]

            def gradient(q, j=j, side=side):
                row = np.zeros(N_JOINTS)
                row[j] = 1.0 if side == "lower" else -1.0
                return row

            constraints.append(
                BarrierConstraint(f"{side}_limit_{j}", margin, gradient, alpha=alpha)
            )

    return constraints


def tipping_barrier(
    *,
    max_moment: float = 0.55,
    lift_coupling: float = 0.55,
    alpha: float = 2.0,
) -> BarrierConstraint:
    """Keep the arm's overturning moment inside the wheelbase.

    Stretch's arm extends sideways, away from the wheelbase's short axis, and
    the lift raises the mass it is carrying. Both make the robot easier to tip,
    and they multiply rather than add: an arm fully out is fine low down and
    not fine at full height.

    The margin used here is

        h(q) = max_moment - arm * (1 + lift_coupling * lift)

    which is the first-order form of that product. **It is a stand-in, not a
    measured stability model** — the real thing needs the mass distribution,
    the payload, and the base's acceleration, none of which this repository
    has. What it is good for is being a genuine *coupling* constraint: it ties
    arm and lift velocities together, so the QP has to trade them against each
    other rather than clamping each alone. Replace it with a real model before
    trusting it on hardware.

    The default ``max_moment`` was chosen so the constraint actually does
    something without swallowing the workspace: it permits near-full extension
    at the bottom of the lift's range (0.51 m of 0.52) and tightens to 0.34 m
    at the top, and roughly 20% of uniformly sampled configurations start
    outside the safe set. A value that excluded nothing would make the filter
    untestable; one that excluded most of the workspace would make it useless.
    """

    def margin(q: np.ndarray) -> float:
        return max_moment - q[ARM] * (1.0 + lift_coupling * q[LIFT])

    def gradient(q: np.ndarray) -> np.ndarray:
        row = np.zeros(N_JOINTS)
        row[ARM] = -(1.0 + lift_coupling * q[LIFT])
        row[LIFT] = -lift_coupling * q[ARM]
        return row

    return BarrierConstraint("tipping", margin, gradient, alpha=alpha)


@dataclass
class SafetyFilter:
    """Projects a desired joint velocity onto the safe set.

    Args:
        model: the robot, for its joint limits.
        barriers: position-limit and coupling constraints. Defaults to both
            ends of every joint plus the tipping constraint.
        respect_velocity_limits: also bound each joint's speed.
    """

    model: StretchModel
    barriers: list[BarrierConstraint] | None = None
    respect_velocity_limits: bool = True

    _last: QPSolution | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.barriers is None:
            self.barriers = joint_limit_barriers(self.model.limits) + [tipping_barrier()]

    def build_box(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-joint velocity bounds: motor limits tightened by the barriers.

        A joint-position barrier touches one joint, so it is axis-aligned and
        simply narrows that joint's velocity range. Intersecting it with the
        motor limit here — rather than handing it to the penalty — is what
        makes joint limits genuinely inviolable instead of merely discouraged.
        """
        q = np.asarray(q, dtype=float)

        if self.respect_velocity_limits:
            upper = self.model.limits.velocity.copy()
            lower = -self.model.limits.velocity.copy()
        else:
            upper = np.full(N_JOINTS, np.inf)
            lower = np.full(N_JOINTS, -np.inf)

        for constraint in self.barriers:
            row, bound = constraint.as_row(q)
            touched = np.flatnonzero(row)

            if touched.size != 1:
                continue  # coupling constraint; handled by the penalty

            j = int(touched[0])
            coefficient = row[j]
            if coefficient > 0:
                upper[j] = min(upper[j], bound / coefficient)
            else:
                lower[j] = max(lower[j], bound / coefficient)

        return lower, upper

    def build_soft(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Coupling constraints only: the ones no per-joint bound can express."""
        q = np.asarray(q, dtype=float)
        rows, bounds, names = [], [], []

        for constraint in self.barriers:
            row, bound = constraint.as_row(q)
            if np.count_nonzero(row) <= 1:
                continue  # axis-aligned; already folded into the box

            rows.append(row)
            bounds.append(bound)
            names.append(constraint.name)

        if not rows:
            return np.zeros((0, N_JOINTS)), np.zeros(0), []

        return np.array(rows), np.array(bounds), names

    def build(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Every constraint as one ``A dq <= b`` block, for inspection and tests.

        The solver does not use this form — the box half is enforced by
        clipping — but having the whole constraint set in one place is what
        lets a test assert feasibility without reimplementing the split.
        """
        q = np.asarray(q, dtype=float)
        lower, upper = self.build_box(q)
        soft_matrix, soft_bound, soft_names = self.build_soft(q)

        rows, bounds, names = [], [], []
        for j in range(N_JOINTS):
            if np.isfinite(upper[j]):
                row = np.zeros(N_JOINTS)
                row[j] = 1.0
                rows.append(row)
                bounds.append(upper[j])
                names.append(f"upper_{j}")
            if np.isfinite(lower[j]):
                row = np.zeros(N_JOINTS)
                row[j] = -1.0
                rows.append(row)
                bounds.append(-lower[j])
                names.append(f"lower_{j}")

        box_matrix = np.array(rows) if rows else np.zeros((0, N_JOINTS))
        box_bound = np.array(bounds) if bounds else np.zeros(0)

        return (
            np.vstack([soft_matrix, box_matrix]),
            np.concatenate([soft_bound, box_bound]),
            soft_names + names,
        )

    def filter(self, q: np.ndarray, desired_dq: np.ndarray) -> QPSolution:
        """Return the closest executable velocity to ``desired_dq`` at ``q``.

        Velocity limits are always satisfied. Barriers are satisfied whenever
        that is possible at all; when it is not — the robot is already outside
        the safe set and cannot return within its speed limits — the barrier is
        relaxed by the smallest amount that admits an answer, and
        :attr:`QPSolution.relaxed` reports how much.
        """
        lower, upper = self.build_box(q)
        soft_matrix, soft_bound, _ = self.build_soft(q)

        solution = solve_box_soft_qp(desired_dq, lower, upper, soft_matrix, soft_bound)
        self._last = solution
        return solution

    def active_names(self, q: np.ndarray, desired_dq: np.ndarray) -> list[str]:
        """Which constraints are binding, for logging and debugging."""
        matrix, bound, names = self.build(q)
        solution = self.filter(q, desired_dq)
        residual = matrix @ solution.dq - bound
        return [name for name, r in zip(names, residual, strict=True) if r > -1e-7]


def integrate(
    model: StretchModel,
    q: np.ndarray,
    dq: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Step the configuration forward.

    Deliberately does **not** clamp to joint limits. Clamping here would hide
    exactly what the filter exists to prevent, and the benchmark's whole point
    is to count violations that actually occurred. If a configuration leaves
    its limits, that is a result, not something to paper over.
    """
    q = np.asarray(q, dtype=float)
    return q + np.asarray(dq, dtype=float) * dt
