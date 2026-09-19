"""Tests for the QP and the safety filter.

The QP is tested against properties that must hold for *any* input — it is a
projection, so idempotence, feasibility and optimality are all checkable
without knowing the answer in advance. The filter is tested against the
situations it exists for: a joint driven into its limit, and a coupled
constraint that no per-joint clamp could enforce.
"""

import numpy as np
import pytest

from stretchwbc.model import ARM, LIFT, N_JOINTS, StretchModel
from stretchwbc.safety import (
    SafetyFilter,
    integrate,
    joint_limit_barriers,
    solve_box_soft_qp,
    solve_projection_qp,
    tipping_barrier,
)


@pytest.fixture
def model():
    return StretchModel()


@pytest.fixture
def home(model):
    return (model.limits.lower + model.limits.upper) / 2.0


# ------------------------------------------------------------------- the QP


def test_feasible_request_passes_through_untouched():
    """The filter must be invisible when nothing is binding."""
    desired = np.array([0.3, -0.2])
    solution = solve_projection_qp(desired, np.eye(2), np.array([1.0, 1.0]))

    assert solution.dq == pytest.approx(desired)
    assert not solution.modified
    assert solution.active == 0


def test_projects_onto_a_half_space():
    desired = np.array([0.3, -0.2])
    solution = solve_projection_qp(desired, np.array([[1.0, 0.0]]), np.array([0.1]))

    assert solution.dq == pytest.approx([0.1, -0.2])


def test_projection_is_the_nearest_feasible_point():
    """Checked against a brute-force search over the feasible set."""
    rng = np.random.default_rng(0)
    desired = np.array([0.8, 0.6])

    matrix = np.array([[1.0, 1.0], [-1.0, 0.4]])
    bound = np.array([0.5, 0.2])

    solution = solve_projection_qp(desired, matrix, bound)
    assert np.all(matrix @ solution.dq <= bound + 1e-8)

    candidates = rng.uniform(-2.0, 2.0, size=(200_000, 2))
    feasible = candidates[np.all(candidates @ matrix.T <= bound, axis=1)]
    best = float(np.min(np.linalg.norm(feasible - desired, axis=1)))

    assert float(np.linalg.norm(solution.dq - desired)) <= best + 1e-3


def test_projection_is_idempotent():
    """Projecting an already-projected point must change nothing."""
    desired = np.array([2.0, -1.5, 0.7])
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    bound = np.array([0.4, -0.2])

    once = solve_projection_qp(desired, matrix, bound).dq
    twice = solve_projection_qp(once, matrix, bound).dq

    assert twice == pytest.approx(once, abs=1e-8)


def test_handles_redundant_duplicate_constraints():
    """Two identical rows must not destabilise the dual iteration."""
    desired = np.array([1.0, 1.0])
    matrix = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    bound = np.array([0.25, 0.25, 0.25])

    solution = solve_projection_qp(desired, matrix, bound)

    assert solution.dq == pytest.approx([0.25, 1.0], abs=1e-8)
    assert solution.converged


def test_handles_a_zero_row():
    """A row of zeros carries no information and must be skipped, not divided by."""
    desired = np.array([1.0, 1.0])
    matrix = np.array([[0.0, 0.0], [1.0, 0.0]])
    bound = np.array([1.0, 0.3])

    solution = solve_projection_qp(desired, matrix, bound)

    assert np.all(np.isfinite(solution.dq))
    assert solution.dq[0] == pytest.approx(0.3, abs=1e-8)


def test_no_constraints_is_the_identity():
    desired = np.array([1.0, -2.0])
    solution = solve_projection_qp(desired, np.zeros((0, 2)), np.zeros(0))

    assert solution.dq == pytest.approx(desired)
    assert solution.converged


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="row count"):
        solve_projection_qp(np.zeros(2), np.zeros((3, 2)), np.zeros(2))

    with pytest.raises(ValueError, match="columns"):
        solve_projection_qp(np.zeros(2), np.zeros((1, 5)), np.zeros(1))


# --------------------------------------------------------------- barriers


def test_joint_limit_barriers_come_in_pairs(model):
    barriers = joint_limit_barriers(model.limits)
    assert len(barriers) == 2 * N_JOINTS


def test_barrier_margin_is_positive_inside(model, home):
    for barrier in joint_limit_barriers(model.limits):
        assert barrier.margin(home) > 0.0


def test_tipping_margin_shrinks_as_the_lift_rises(model):
    """The coupling, stated as a test: height costs reach."""
    barrier = tipping_barrier()

    low = np.zeros(N_JOINTS)
    low[LIFT], low[ARM] = 0.2, 0.35

    high = low.copy()
    high[LIFT] = 1.0

    assert barrier.margin(high) < barrier.margin(low)


def test_tipping_gradient_touches_both_joints(model):
    barrier = tipping_barrier()
    q = np.zeros(N_JOINTS)
    q[LIFT], q[ARM] = 0.7, 0.3

    gradient = barrier.gradient(q)

    assert gradient[ARM] != 0.0
    assert gradient[LIFT] != 0.0
    assert np.count_nonzero(gradient) == 2


# ----------------------------------------------------------------- filter


def test_zero_command_stays_zero(model, home):
    solution = SafetyFilter(model).filter(home, np.zeros(N_JOINTS))

    assert solution.dq == pytest.approx(np.zeros(N_JOINTS), abs=1e-9)
    assert solution.converged


def test_modest_command_in_the_interior_is_untouched(model, home):
    desired = np.zeros(N_JOINTS)
    desired[0] = 0.05

    solution = SafetyFilter(model).filter(home, desired)

    assert solution.dq == pytest.approx(desired, abs=1e-8)
    assert not solution.modified


def test_velocity_limits_are_enforced(model, home):
    filt = SafetyFilter(model)
    desired = np.full(N_JOINTS, 50.0)

    solution = filt.filter(home, desired)

    assert np.all(np.abs(solution.dq) <= model.limits.velocity + 1e-7)


def test_a_joint_at_its_limit_cannot_be_pushed_further(model):
    """The barrier goes to zero at the boundary, so the approach velocity does too."""
    filt = SafetyFilter(model)

    q = (model.limits.lower + model.limits.upper) / 2.0
    q[LIFT] = model.limits.upper[LIFT] - 1e-4  # right at the top of the lift

    desired = np.zeros(N_JOINTS)
    desired[LIFT] = 1.0

    solution = filt.filter(q, desired)

    assert solution.dq[LIFT] < 1e-3


def test_a_joint_near_its_limit_is_slowed_not_stopped(model):
    """The point of a barrier rather than a hard stop: it decelerates."""
    filt = SafetyFilter(model)
    desired = np.zeros(N_JOINTS)
    desired[LIFT] = 0.1

    far = (model.limits.lower + model.limits.upper) / 2.0
    near = far.copy()
    near[LIFT] = model.limits.upper[LIFT] - 0.02

    assert filt.filter(near, desired).dq[LIFT] < filt.filter(far, desired).dq[LIFT]


def test_simulation_never_leaves_the_joint_limits(model):
    """The property the whole filter exists for.

    A hostile controller asking for large random velocities from random starts,
    integrated without any clamping, must never produce a configuration outside
    its limits.
    """
    rng = np.random.default_rng(1)
    filt = SafetyFilter(model)
    dt = 0.05

    violations = 0
    for _ in range(40):
        q = model.random_configuration(rng)
        q = model.limits.clamp(q)

        for _ in range(60):
            desired = rng.uniform(-3.0, 3.0, size=N_JOINTS)
            dq = filt.filter(q, desired).dq
            q = integrate(model, q, dq, dt)

            if not model.limits.within(q, tolerance=1e-4):
                violations += 1

    assert violations == 0


def test_tipping_is_enforced_where_a_per_joint_clamp_could_not(model):
    """The reason this is a QP and not eight independent clamps.

    Both the arm and the lift are comfortably inside their own limits, so no
    per-joint rule would object. The combination is what is unsafe, and only a
    constraint that sees both can refuse it.
    """
    filt = SafetyFilter(model)

    q = (model.limits.lower + model.limits.upper) / 2.0
    q[ARM] = 0.36  # limit is 0.52
    q[LIFT] = 0.75  # limits are 0.15 to 1.10

    assert model.limits.within(q)
    assert model.limits.violations(q).sum() == 0.0

    desired = np.zeros(N_JOINTS)
    desired[ARM] = 0.15
    desired[LIFT] = 0.15

    solution = filt.filter(q, desired)

    assert solution.modified
    assert "tipping" in filt.active_names(q, desired)
    assert solution.dq[ARM] < desired[ARM]


def test_active_names_reports_nothing_in_the_interior(model, home):
    filt = SafetyFilter(model)
    desired = np.zeros(N_JOINTS)
    desired[0] = 0.01

    assert filt.active_names(home, desired) == []


def test_integrate_does_not_clamp(model):
    """Clamping inside integrate would hide the violations being measured."""
    q = model.limits.upper.copy()
    dq = np.ones(N_JOINTS)

    stepped = integrate(model, q, dq, 1.0)

    assert np.all(stepped > model.limits.upper)


# ------------------------------------------------ hard box, soft barriers


def test_box_is_never_violated_even_after_one_iteration():
    """Hard feasibility here is structural, not something the solver achieves.

    Every iterate is clipped into the box, so stopping the solver arbitrarily
    early costs accuracy against the barriers and nothing at all against the
    velocity limits. This is the property that a dual method could not offer,
    and its absence was a real bug: the first version of this filter returned
    commands exceeding the velocity limits by over 0.1 rad/s whenever the dual
    failed to converge.
    """
    rng = np.random.default_rng(0)
    box = np.array([0.4, 0.2, 0.9])

    for _ in range(200):
        desired = rng.uniform(-5.0, 5.0, size=3)
        soft_matrix = rng.normal(size=(4, 3))
        soft_bound = rng.uniform(-1.0, 1.0, size=4)

        for cap in (1, 2, 5, 50):
            solution = solve_box_soft_qp(
                desired, -box, box, soft_matrix, soft_bound, max_iterations=cap
            )
            assert np.all(np.abs(solution.dq) <= box + 1e-12)


def test_box_soft_matches_a_plain_clip_when_no_soft_constraints():
    desired = np.array([3.0, -0.1, 0.05])
    box = np.array([1.0, 1.0, 1.0])

    solution = solve_box_soft_qp(desired, -box, box, np.zeros((0, 3)), np.zeros(0))

    assert solution.dq == pytest.approx([1.0, -0.1, 0.05])
    assert solution.converged
    assert solution.relaxed == 0.0


def test_soft_constraint_is_satisfied_when_it_can_be():
    """A feasible soft constraint should end up with essentially no slack."""
    desired = np.array([1.0, 1.0])
    box = np.array([2.0, 2.0])
    soft_matrix = np.array([[1.0, 1.0]])
    soft_bound = np.array([0.5])

    solution = solve_box_soft_qp(desired, -box, box, soft_matrix, soft_bound)

    assert solution.relaxed < 1e-3
    assert float((soft_matrix @ solution.dq)[0]) <= 0.5 + 1e-3


def test_infeasible_soft_constraint_reports_its_slack():
    """The case that broke the dual formulation.

    The box confines dq to [0.5, 0.5] at most, while the soft constraint
    demands the sum be below -3. Nothing inside the box can do that. The filter
    must still return something executable and say how far short it fell,
    rather than returning an out-of-box command that looks fine.
    """
    desired = np.array([1.0, 1.0])
    box = np.array([0.5, 0.5])
    soft_matrix = np.array([[1.0, 1.0]])
    soft_bound = np.array([-3.0])

    solution = solve_box_soft_qp(desired, -box, box, soft_matrix, soft_bound)

    assert np.all(np.abs(solution.dq) <= box + 1e-12)
    assert solution.relaxed > 1.0


def test_higher_penalty_tracks_the_soft_constraint_more_closely():
    desired = np.array([2.0, 0.0])
    box = np.array([2.0, 2.0])
    soft_matrix = np.array([[1.0, 0.0]])
    soft_bound = np.array([0.4])

    loose = solve_box_soft_qp(desired, -box, box, soft_matrix, soft_bound, penalty=5.0)
    tight = solve_box_soft_qp(desired, -box, box, soft_matrix, soft_bound, penalty=5.0e3)

    assert tight.relaxed < loose.relaxed


def test_box_soft_validates_its_inputs():
    with pytest.raises(ValueError, match="one entry per variable"):
        solve_box_soft_qp(np.zeros(3), -np.ones(2), np.ones(2), np.zeros((0, 3)), np.zeros(0))

    with pytest.raises(ValueError, match="row count"):
        solve_box_soft_qp(np.zeros(2), -np.ones(2), np.ones(2), np.zeros((3, 2)), np.zeros(2))


def test_inverted_box_collapses_rather_than_raising():
    """A joint already outside its limit can invert the box.

    A safety filter that refuses to answer is not a useful safety filter, so
    the box collapses to its midpoint and the call still returns something
    executable.
    """
    solution = solve_box_soft_qp(
        np.array([5.0]),
        np.array([0.4]),
        np.array([-0.2]),
        np.zeros((0, 1)),
        np.zeros(0),
    )

    assert np.isfinite(solution.dq).all()
    assert solution.dq[0] == pytest.approx(0.1)


def test_filter_never_exceeds_velocity_limits_under_hostile_input(model):
    """The end-to-end version of the same guarantee, on the real robot model."""
    rng = np.random.default_rng(7)
    filt = SafetyFilter(model)

    for _ in range(300):
        q = model.limits.clamp(model.random_configuration(rng))
        desired = rng.uniform(-8.0, 8.0, size=N_JOINTS)

        dq = filt.filter(q, desired).dq
        assert np.all(np.abs(dq) <= model.limits.velocity + 1e-9)


def test_filter_reports_relaxation_only_outside_the_safe_set(model):
    """Inside the tipping-safe set a modest request needs no relaxation."""
    filt = SafetyFilter(model)

    inside = (model.limits.lower + model.limits.upper) / 2.0
    inside[ARM] = 0.10
    inside[LIFT] = 0.30

    desired = np.zeros(N_JOINTS)
    desired[ARM] = 0.02

    assert filt.filter(inside, desired).relaxed < 1e-6


def test_axis_aligned_barriers_are_folded_into_the_box(model, home):
    """Joint-limit barriers must narrow the box, not join the penalty."""
    filt = SafetyFilter(model)

    q = home.copy()
    q[LIFT] = model.limits.upper[LIFT] - 0.01

    _, upper = filt.build_box(q)
    assert upper[LIFT] < model.limits.velocity[LIFT]

    # Only genuinely coupling rows should be left for the penalty.
    soft_matrix, _, names = filt.build_soft(q)
    assert names == ["tipping"]
    assert np.count_nonzero(soft_matrix[0]) == 2


def test_persistent_pressure_never_breaches_a_joint_limit(model):
    """The case that exposed the original design as wrong.

    The hostile controller in the earlier test picks a *new* random velocity
    every step, so it jitters and rarely leans on any one limit. A learned
    policy does the opposite: it pushes the same direction for many steps. With
    the joint barriers in the soft set, that persistent pressure — in conflict
    with the coupling constraint — pushed joints past their limits 132 times
    across 50 rollouts. Folding the axis-aligned barriers into the box took it
    to zero.
    """
    rng = np.random.default_rng(1)
    filt = SafetyFilter(model)

    violations = 0
    for _ in range(40):
        q = model.limits.clamp(model.random_configuration(rng))
        push = rng.uniform(-3.0, 3.0, size=N_JOINTS)  # same command every step

        for _ in range(60):
            q = integrate(model, q, filt.filter(q, push).dq, 0.05)
            if not model.limits.within(q, tolerance=1e-6):
                violations += 1

    assert violations == 0
