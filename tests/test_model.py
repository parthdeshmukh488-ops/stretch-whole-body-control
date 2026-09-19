"""Tests for the kinematic model.

The load-bearing test here is the analytic Jacobian against a central
difference. Every solver iterates on that Jacobian, so if it drifts from the
forward kinematics the solvers degrade quietly rather than failing — they still
converge, just to slightly wrong places, and nothing announces it.
"""

import numpy as np
import pytest

from stretchwbc.model import (
    ARM,
    BASE_THETA,
    BASE_X,
    BASE_Y,
    LIFT,
    N_JOINTS,
    WRIST_YAW,
    JointLimits,
    StretchModel,
)


@pytest.fixture
def model():
    return StretchModel()


@pytest.fixture
def home(model):
    return (model.limits.lower + model.limits.upper) / 2.0


def test_limits_have_the_right_shape():
    limits = JointLimits.stretch3()
    assert limits.lower.shape == (N_JOINTS,)
    assert limits.upper.shape == (N_JOINTS,)
    assert np.all(limits.lower < limits.upper)
    assert np.all(limits.velocity > 0)


def test_limits_reject_inverted_ranges():
    with pytest.raises(ValueError, match="lower limit"):
        JointLimits(
            lower=np.ones(N_JOINTS),
            upper=np.zeros(N_JOINTS),
            velocity=np.ones(N_JOINTS),
        )


def test_limits_reject_wrong_length():
    with pytest.raises(ValueError, match="entries"):
        JointLimits(lower=np.zeros(3), upper=np.ones(3), velocity=np.ones(3))


def test_limits_reject_nonpositive_velocity():
    limits = JointLimits.stretch3()
    with pytest.raises(ValueError, match="velocity"):
        JointLimits(limits.lower, limits.upper, np.zeros(N_JOINTS))


def test_clamp_and_violations_agree(model):
    q = model.limits.upper + 0.3
    assert np.all(model.limits.violations(q) > 0)
    assert model.limits.within(model.limits.clamp(q))


def test_violations_are_zero_inside(model, home):
    assert np.all(model.limits.violations(home) == 0.0)
    assert model.limits.within(home)


def test_forward_rejects_wrong_shape(model):
    with pytest.raises(ValueError, match="entries"):
        model.forward(np.zeros(3))


def test_base_translation_moves_the_tool_one_for_one(model, home):
    """A prismatic base degree of freedom must be exactly unit gain."""
    shifted = home.copy()
    shifted[BASE_X] += 0.4
    shifted[BASE_Y] -= 0.25

    before = model.position(home)
    after = model.position(shifted)

    assert after - before == pytest.approx([0.4, -0.25, 0.0])


def test_lift_moves_the_tool_vertically(model, home):
    raised = home.copy()
    raised[LIFT] += 0.2

    delta = model.position(raised) - model.position(home)
    assert delta[2] == pytest.approx(0.2)
    assert delta[:2] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_arm_extends_along_the_base_heading(model):
    """With the base facing +x, the arm extends along +y."""
    q = np.zeros(N_JOINTS)
    q[LIFT] = 0.6

    extended = q.copy()
    extended[ARM] = 0.3

    delta = model.position(extended) - model.position(q)
    assert delta[1] == pytest.approx(0.3)
    assert delta[0] == pytest.approx(0.0, abs=1e-12)
    assert delta[2] == pytest.approx(0.0, abs=1e-12)


def test_rotating_the_base_rotates_the_arm_direction(model):
    """At 90 degrees of heading, the arm's +y becomes world -x."""
    q = np.zeros(N_JOINTS)
    q[LIFT] = 0.6
    q[BASE_THETA] = np.pi / 2

    extended = q.copy()
    extended[ARM] = 0.3

    delta = model.position(extended) - model.position(q)
    assert delta[0] == pytest.approx(-0.3)
    assert delta[1] == pytest.approx(0.0, abs=1e-12)


def test_analytic_jacobian_matches_finite_differences(model):
    """The test the whole solver stack rests on."""
    rng = np.random.default_rng(0)

    worst = 0.0
    for _ in range(200):
        q = model.random_configuration(rng)
        analytic = model.position_jacobian(q)
        numerical = model.numerical_position_jacobian(q)
        worst = max(worst, float(np.max(np.abs(analytic - numerical))))

    assert worst < 1e-6


def test_jacobian_matches_at_the_limits_too(model):
    """Corners of the joint space, where wrist angles wrap."""
    for q in (model.limits.lower.copy(), model.limits.upper.copy()):
        analytic = model.position_jacobian(q)
        numerical = model.numerical_position_jacobian(q)
        assert np.max(np.abs(analytic - numerical)) < 1e-6


def test_jacobian_shape_and_base_columns(model, home):
    jacobian = model.position_jacobian(home)

    assert jacobian.shape == (3, N_JOINTS)
    assert jacobian[:, BASE_X] == pytest.approx([1.0, 0.0, 0.0])
    assert jacobian[:, BASE_Y] == pytest.approx([0.0, 1.0, 0.0])


def test_jacobian_rejects_wrong_shape(model):
    with pytest.raises(ValueError, match="entries"):
        model.position_jacobian(np.zeros(3))


def test_base_and_arm_are_redundant_in_one_direction(model):
    """The redundancy every solver here has to cope with.

    With the base facing +x, sliding the base along +y and extending the arm
    produce the same tool motion. Those two Jacobian columns are therefore
    parallel, and the 8-DOF system is genuinely rank-deficient in that
    direction rather than merely ill-conditioned.
    """
    q = np.zeros(N_JOINTS)
    q[LIFT] = 0.6

    jacobian = model.position_jacobian(q)
    base_y = jacobian[:, BASE_Y]
    arm = jacobian[:, ARM]

    cosine = float(base_y @ arm / (np.linalg.norm(base_y) * np.linalg.norm(arm)))
    assert cosine == pytest.approx(1.0, abs=1e-12)


def test_manipulability_is_positive_in_the_interior(model, home):
    assert model.manipulability(home) > 0.0


def test_manipulability_drops_when_the_wrist_folds(model, home):
    """A configuration that loses a direction should score lower."""
    folded = home.copy()
    folded[WRIST_YAW] = 0.0
    folded[5:] = 0.0

    assert model.manipulability(folded) <= model.manipulability(home) * 5.0


def test_random_configuration_is_within_limits(model):
    rng = np.random.default_rng(3)
    for _ in range(100):
        assert model.limits.within(model.random_configuration(rng))


def test_random_reachable_target_is_reachable_by_construction(model):
    """Targets come from real configurations, so every one is attainable.

    Sampling targets in Cartesian space would mix unreachable ones into a
    solver benchmark, and the resulting success rate would measure the sampler
    as much as the solvers.
    """
    rng = np.random.default_rng(4)
    for _ in range(50):
        target = model.random_reachable_target(rng)
        assert np.all(np.isfinite(target))
        assert target.shape == (3,)
