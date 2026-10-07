// The C++ controller against the Python reference, and in closed loop.

#include <gtest/gtest.h>

#include <random>

#include "controller_golden.hpp"
#include "stretch_wbc_ros/controller.hpp"

using stretch_wbc_ros::configure_filter;
using stretch_wbc_ros::Jacobian;
using stretch_wbc_ros::jacobian_transpose_step;
using stretch_wbc_ros::JointVector;
using stretch_wbc_ros::Limits;
using stretch_wbc_ros::StretchKinematics;
using stretch_wbc_ros::tipping_margin;

namespace {

template <std::size_t N>
Eigen::Matrix<double, N, 1> vec(const std::array<double, N>& a) {
  return Eigen::Map<const Eigen::Matrix<double, N, 1>>(a.data());
}

}  // namespace

// Golden values carry 12 significant digits, so agreement is checked to 1e-9.
TEST(Golden, KinematicsMatchPython) {
  const StretchKinematics kin;
  for (const auto& c : kGoldenCases) {
    const JointVector q = vec(c.q);
    EXPECT_LT((kin.position(q) - vec(c.position)).norm(), 1e-9);
    const Jacobian expected =
        Eigen::Map<const Eigen::Matrix<double, 3, 8, Eigen::RowMajor>>(c.jacobian.data());
    EXPECT_LT((kin.jacobian(q) - expected).norm(), 1e-9);
  }
}

TEST(Golden, JacobianTransposeStepMatchesPython) {
  const StretchKinematics kin;
  for (const auto& c : kGoldenCases) {
    const JointVector step = jacobian_transpose_step(kin, vec(c.q), vec(c.target));
    EXPECT_LT((step - vec(c.jt_step)).norm(), 1e-9);
  }
}

TEST(Golden, FilterBoxAndTippingMatchPython) {
  const Limits limits = Limits::stretch3();
  stretchwbc::SafetyFilter filter(limits.velocity, 1);
  for (const auto& c : kGoldenCases) {
    const JointVector q = vec(c.q);
    configure_filter(filter, limits, q);
    // The C++ filter collapses an inverted bound to its midpoint; compare where the
    // Python box is not inverted.
    for (int j = 0; j < 8; ++j) {
      if (c.box_lower[j] <= c.box_upper[j]) {
        EXPECT_NEAR(filter.box_lower()(j), c.box_lower[j], 1e-9);
        EXPECT_NEAR(filter.box_upper()(j), c.box_upper[j], 1e-9);
      }
    }
    EXPECT_NEAR(tipping_margin(q), c.tipping_margin, 1e-9);
  }
}

TEST(Kinematics, JacobianMatchesFiniteDifferences) {
  const StretchKinematics kin;
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (int trial = 0; trial < 20; ++trial) {
    JointVector q;
    for (int j = 0; j < 8; ++j) q(j) = u(rng);
    q(3) = 0.6 + 0.3 * u(rng);
    q(4) = 0.25 + 0.2 * u(rng);
    Jacobian numeric;
    for (int j = 0; j < 8; ++j) {
      JointVector plus = q, minus = q;
      plus(j) += 1e-6;
      minus(j) -= 1e-6;
      numeric.col(j) = (kin.position(plus) - kin.position(minus)) / 2e-6;
    }
    EXPECT_LT((kin.jacobian(q) - numeric).norm(), 1e-6);
  }
}

// The node's control loop, without ROS: step, filter, integrate. It must reach a
// reachable target and never leave the joint limits on the way.
TEST(ClosedLoop, ReachesTargetInsideLimits) {
  const StretchKinematics kin;
  const Limits limits = Limits::stretch3();
  stretchwbc::SafetyFilter filter(limits.velocity, 1);
  const double dt = 0.02;

  JointVector goal;
  goal << 0.4, -0.3, 0.5, 0.9, 0.2, 0.3, -0.2, 0.1;  // inside limits and tipping-safe
  const Eigen::Vector3d target = kin.position(goal);

  JointVector q = JointVector::Zero();
  q(3) = 0.6;
  double error = (target - kin.position(q)).norm();
  for (int tick = 0; tick < 3000 && error > 1e-3; ++tick) {
    configure_filter(filter, limits, q);
    const JointVector velocity = filter.solve(jacobian_transpose_step(kin, q, target) / dt).dq;
    EXPECT_TRUE(((velocity.array().abs() - limits.velocity.array()) <= 1e-9).all());
    q += velocity * dt;
    EXPECT_TRUE(((q - limits.lower).array() >= -1e-9).all());
    EXPECT_TRUE(((limits.upper - q).array() >= -1e-9).all());
    error = (target - kin.position(q)).norm();
  }
  EXPECT_LT(error, 1e-3);
}
