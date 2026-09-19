// Safety filter, ported to C++ for the real-time side of a ROS 2 stack.
//
// The Python implementation in src/stretchwbc/safety.py is the reference: it
// is where the design is worked out and where the test suite is richest. This
// port exists because the filter belongs in the control loop, and the control
// loop should not be holding the GIL or waiting on a Python allocator.
//
// The two implementations are kept in step by a shared set of numeric cases —
// see cpp/test/test_safety_filter.cpp, whose expected values are generated
// from the Python side. A port that drifts from its reference is worse than no
// port, because both look correct in isolation.
//
// Design notes that carry over unchanged:
//
//   * Every AXIS-ALIGNED constraint is a box bound, enforced by clipping each
//     iterate. That covers the motor velocity limits and the joint-position
//     barriers, which touch one joint each. The returned command is therefore
//     executable even if the solver is stopped early -- the property that
//     makes this safe to run under a deadline.
//   * Only genuinely COUPLING constraints are soft. The tipping constraint
//     ties arm extension to lift height, and can be unsatisfiable when the
//     robot is already outside the safe set, so treating it as hard would ask
//     for something that does not exist.
//
// Note on allocation: solve() does allocate, for the copies of the soft block
// it works on. Making it allocation-free is a worthwhile change for a hard
// real-time thread and is not done here, because the version that avoided the
// copies was harder to read and this has not yet been profiled in a real loop.
// Measure before optimising it.

#pragma once

#include <Eigen/Dense>
#include <stdexcept>

namespace stretchwbc {

inline constexpr int kNumJoints = 8;

using JointVector = Eigen::Matrix<double, kNumJoints, 1>;

/// Outcome of one filter step.
struct FilterResult {
  JointVector dq;          ///< command to send; always inside the velocity box
  double correction{0.0};  ///< how far the request was moved, rad/s
  double relaxed{0.0};     ///< largest soft-constraint violation left over
  int iterations{0};
  bool converged{false};

  [[nodiscard]] bool modified() const { return correction > 1e-12; }
};

/// Parameters of the accelerated projected-gradient solver.
struct SolverOptions {
  double penalty{2.0e3};   ///< weight on soft-constraint violation
  int max_iterations{600};
  double tolerance{1e-11};
};

/// Projects a desired joint velocity onto a box, penalising soft constraints.
///
/// Solves
///     min  0.5||dq - desired||^2 + 0.5 * penalty * sum_i max(0, a_i.dq - b_i)^2
///     s.t. |dq_j| <= box_j
///
/// The box constraint is satisfied exactly at every iterate. The soft rows are
/// whatever the caller supplies — joint-limit barriers, a tipping constraint,
/// self-collision margins.
class SafetyFilter {
 public:
  /// @param velocity_limit  per-joint speed bound; must be positive. Used as
  ///                        the default box until set_box() narrows it.
  /// @param max_soft_rows   capacity reserved for coupling constraints.
  SafetyFilter(const JointVector& velocity_limit, int max_soft_rows,
               SolverOptions options = {})
      : velocity_limit_(velocity_limit),
        box_lower_(-velocity_limit),
        box_upper_(velocity_limit),
        options_(options),
        soft_matrix_(Eigen::MatrixXd::Zero(max_soft_rows, kNumJoints)),
        soft_bound_(Eigen::VectorXd::Zero(max_soft_rows)),
        max_soft_rows_(max_soft_rows) {
    if ((velocity_limit_.array() <= 0.0).any()) {
      throw std::invalid_argument("velocity limits must be positive");
    }
    if (max_soft_rows < 0) {
      throw std::invalid_argument("max_soft_rows must be non-negative");
    }
  }

  /// Replace the soft constraint block. Must fit within the reserved capacity.
  void set_soft_constraints(const Eigen::MatrixXd& matrix,
                            const Eigen::VectorXd& bound) {
    if (matrix.rows() != bound.size()) {
      throw std::invalid_argument("soft matrix and bound disagree on row count");
    }
    if (matrix.rows() > max_soft_rows_) {
      throw std::invalid_argument("more soft rows than reserved capacity");
    }
    if (matrix.rows() > 0 && matrix.cols() != kNumJoints) {
      throw std::invalid_argument("soft matrix must have kNumJoints columns");
    }
    active_soft_rows_ = static_cast<int>(matrix.rows());
    soft_matrix_.topRows(active_soft_rows_) = matrix;
    soft_bound_.head(active_soft_rows_) = bound;
    lipschitz_dirty_ = true;
  }

  /// Narrow the per-joint velocity box, as the joint-position barriers do.
  ///
  /// An inverted bound -- which a barrier produces when a joint is already
  /// outside its position limit -- collapses to its midpoint rather than
  /// throwing. A safety filter that refuses to answer is not useful.
  void set_box(const JointVector& lower, const JointVector& upper) {
    box_lower_ = lower;
    box_upper_ = upper;
    for (int i = 0; i < kNumJoints; ++i) {
      if (box_lower_(i) > box_upper_(i)) {
        const double midpoint = 0.5 * (box_lower_(i) + box_upper_(i));
        box_lower_(i) = midpoint;
        box_upper_(i) = midpoint;
      }
    }
  }

  [[nodiscard]] const JointVector& box_lower() const { return box_lower_; }
  [[nodiscard]] const JointVector& box_upper() const { return box_upper_; }

  [[nodiscard]] int soft_row_count() const { return active_soft_rows_; }

  /// Closest executable velocity to `desired`.
  FilterResult solve(const JointVector& desired);

 private:
  [[nodiscard]] JointVector clip(const JointVector& v) const {
    return v.cwiseMax(box_lower_).cwiseMin(box_upper_);
  }

  void refresh_lipschitz();

  JointVector velocity_limit_;
  JointVector box_lower_;
  JointVector box_upper_;
  SolverOptions options_;

  Eigen::MatrixXd soft_matrix_;
  Eigen::VectorXd soft_bound_;

  int max_soft_rows_{0};
  int active_soft_rows_{0};

  double lipschitz_{1.0};
  bool lipschitz_dirty_{true};
};

}  // namespace stretchwbc
