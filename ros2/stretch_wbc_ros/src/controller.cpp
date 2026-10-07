#include "stretch_wbc_ros/controller.hpp"

#include <cmath>

namespace stretch_wbc_ros {
namespace {

// stretchwbc.safety.tipping_barrier defaults
constexpr double kMaxMoment = 0.55;
constexpr double kLiftCoupling = 0.55;
constexpr double kTippingAlpha = 2.0;

Eigen::Matrix3d rot_z(double a) {
  return Eigen::AngleAxisd(a, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}
Eigen::Matrix3d rot_y(double a) {
  return Eigen::AngleAxisd(a, Eigen::Vector3d::UnitY()).toRotationMatrix();
}
Eigen::Matrix3d rot_x(double a) {
  return Eigen::AngleAxisd(a, Eigen::Vector3d::UnitX()).toRotationMatrix();
}

// Derivatives of the elementary rotations with respect to their angle.
Eigen::Matrix3d d_rot_z(double a) {
  Eigen::Matrix3d m;
  m << -std::sin(a), -std::cos(a), 0, std::cos(a), -std::sin(a), 0, 0, 0, 0;
  return m;
}
Eigen::Matrix3d d_rot_y(double a) {
  Eigen::Matrix3d m;
  m << -std::sin(a), 0, std::cos(a), 0, 0, 0, -std::cos(a), 0, -std::sin(a);
  return m;
}
Eigen::Matrix3d d_rot_x(double a) {
  Eigen::Matrix3d m;
  m << 0, 0, 0, 0, -std::sin(a), -std::cos(a), 0, std::cos(a), -std::sin(a);
  return m;
}

}  // namespace

Limits Limits::stretch3(double workspace_m) {
  Limits l;
  l.lower << -workspace_m, -workspace_m, -M_PI, 0.15, 0.0, -1.75, -1.57, -3.14;
  l.upper << workspace_m, workspace_m, M_PI, 1.10, 0.52, 4.00, 0.56, 3.14;
  l.velocity << 0.25, 0.25, 0.60, 0.15, 0.15, 1.50, 1.50, 1.50;
  return l;
}

JointVector Limits::clamp(const JointVector& q) const { return q.cwiseMax(lower).cwiseMin(upper); }

Eigen::Matrix3d StretchKinematics::orientation(const JointVector& q) const {
  return rot_z(q(kBaseTheta)) * rot_z(q(kWristYaw)) * rot_y(q(kWristPitch)) * rot_x(q(kWristRoll));
}

Eigen::Vector3d StretchKinematics::position(const JointVector& q) const {
  const Eigen::Vector3d in_base(mast_offset_x, mast_offset_y + arm_retracted_y + q(kArm), q(kLift));
  const Eigen::Matrix3d wrist = rot_z(q(kWristYaw)) * rot_y(q(kWristPitch)) * rot_x(q(kWristRoll));
  const Eigen::Vector3d tool = wrist * Eigen::Vector3d(wrist_length, 0, 0);
  return Eigen::Vector3d(q(kBaseX), q(kBaseY), 0) + rot_z(q(kBaseTheta)) * (in_base + tool);
}

Jacobian StretchKinematics::jacobian(const JointVector& q) const {
  const double theta = q(kBaseTheta);
  const double yaw = q(kWristYaw), pitch = q(kWristPitch), roll = q(kWristRoll);
  const Eigen::Matrix3d base = rot_z(theta);
  const Eigen::Matrix3d rz = rot_z(yaw), ry = rot_y(pitch), rx = rot_x(roll);
  const Eigen::Vector3d axis(wrist_length, 0, 0);
  const Eigen::Vector3d in_base(mast_offset_x, mast_offset_y + arm_retracted_y + q(kArm), q(kLift));
  const Eigen::Vector3d local = in_base + rz * ry * rx * axis;

  Jacobian j = Jacobian::Zero();
  j.col(kBaseX) = Eigen::Vector3d::UnitX();
  j.col(kBaseY) = Eigen::Vector3d::UnitY();
  j.col(kBaseTheta) = d_rot_z(theta) * local;
  j.col(kLift) = base * Eigen::Vector3d::UnitZ();
  j.col(kArm) = base * Eigen::Vector3d::UnitY();
  j.col(kWristYaw) = base * (d_rot_z(yaw) * ry * rx * axis);
  j.col(kWristPitch) = base * (rz * d_rot_y(pitch) * rx * axis);
  j.col(kWristRoll) = base * (rz * ry * d_rot_x(roll) * axis);
  return j;
}

JointVector jacobian_transpose_step(const StretchKinematics& kinematics, const JointVector& q,
                                    const Eigen::Vector3d& target, double max_step) {
  const Eigen::Vector3d error = target - kinematics.position(q);
  const Jacobian j = kinematics.jacobian(q);
  const JointVector direction = j.transpose() * error;
  const Eigen::Vector3d projected = j * direction;
  const double denominator = projected.squaredNorm();
  if (denominator < 1e-18) {
    return JointVector::Zero();
  }
  JointVector step = (error.dot(projected) / denominator) * direction;
  const double norm = step.norm();
  if (norm > max_step) {
    step *= max_step / norm;
  }
  return step;
}

double tipping_margin(const JointVector& q) {
  return kMaxMoment - q(kArm) * (1.0 + kLiftCoupling * q(kLift));
}

void configure_filter(stretchwbc::SafetyFilter& filter, const Limits& limits, const JointVector& q,
                      double barrier_alpha) {
  // Joint-limit barriers touch one joint each, so they narrow that joint's box.
  const JointVector lower = (-limits.velocity).cwiseMax(-barrier_alpha * (q - limits.lower));
  const JointVector upper = limits.velocity.cwiseMin(barrier_alpha * (limits.upper - q));
  filter.set_box(lower, upper);

  // Tipping couples arm and lift: one soft row  -grad_h . dq <= alpha h.
  Eigen::MatrixXd row = Eigen::MatrixXd::Zero(1, kNumJoints);
  row(0, kArm) = 1.0 + kLiftCoupling * q(kLift);
  row(0, kLift) = kLiftCoupling * q(kArm);
  Eigen::VectorXd bound(1);
  bound(0) = kTippingAlpha * tipping_margin(q);
  filter.set_soft_constraints(row, bound);
}

}  // namespace stretch_wbc_ros
