// Whole-body reaching for the ROS 2 node: kinematics, the IK step and the
// safety-filter set-up, ported from the Python reference.
//
//   StretchKinematics        <- stretchwbc.model.StretchModel (position, Jacobian)
//   jacobian_transpose_step  <- stretchwbc.ik.solve_jacobian_transpose (one iteration)
//   configure_filter         <- stretchwbc.safety.SafetyFilter.build_box + tipping_barrier
//
// test/test_controller.cpp checks every one of these against numbers generated
// by the Python code (tools/generate_ros_golden.py), so the two cannot drift
// apart silently.

#pragma once

#include <Eigen/Dense>

#include "stretchwbc/safety_filter.hpp"

namespace stretch_wbc_ros {

using stretchwbc::JointVector;
using stretchwbc::kNumJoints;
using Jacobian = Eigen::Matrix<double, 3, kNumJoints>;

enum Joint : int { kBaseX, kBaseY, kBaseTheta, kLift, kArm, kWristYaw, kWristPitch, kWristRoll };

struct Limits {
  JointVector lower;
  JointVector upper;
  JointVector velocity;

  /// The Stretch 3 values of stretchwbc.model.JointLimits.stretch3().
  static Limits stretch3(double workspace_m = 3.0);
  [[nodiscard]] JointVector clamp(const JointVector& q) const;
};

/// Simplified Stretch 3 kinematics: the same model and dimensions as the Python
/// StretchModel (see its docstring for what is simplified).
class StretchKinematics {
 public:
  double mast_offset_x{-0.10};
  double mast_offset_y{0.14};
  double arm_retracted_y{0.08};
  double wrist_length{0.22};

  [[nodiscard]] Eigen::Vector3d position(const JointVector& q) const;
  [[nodiscard]] Eigen::Matrix3d orientation(const JointVector& q) const;
  [[nodiscard]] Jacobian jacobian(const JointVector& q) const;
};

/// One Jacobian-transpose step towards `target`: dq = alpha J^T e, with alpha
/// minimising the residual along that direction, and the step norm capped at
/// `max_step`. Returns zero when the direction vanishes (a singular target).
[[nodiscard]] JointVector jacobian_transpose_step(const StretchKinematics& kinematics,
                                                  const JointVector& q,
                                                  const Eigen::Vector3d& target,
                                                  double max_step = 0.05);

/// Load the filter with the constraints for configuration q: the motor velocity
/// limits narrowed by joint-limit barriers (exact box bounds) and the tipping
/// constraint (a soft coupling row).
void configure_filter(stretchwbc::SafetyFilter& filter, const Limits& limits,
                      const JointVector& q, double barrier_alpha = 4.0);

/// Tipping margin of stretchwbc.safety.tipping_barrier: positive means safe.
[[nodiscard]] double tipping_margin(const JointVector& q);

}  // namespace stretch_wbc_ros
