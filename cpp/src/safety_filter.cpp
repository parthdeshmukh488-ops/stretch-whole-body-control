#include "stretchwbc/safety_filter.hpp"

#include <Eigen/SVD>
#include <algorithm>
#include <cmath>

namespace stretchwbc {

void SafetyFilter::refresh_lipschitz() {
  if (!lipschitz_dirty_) {
    return;
  }

  if (active_soft_rows_ == 0) {
    lipschitz_ = 1.0;
    lipschitz_dirty_ = false;
    return;
  }

  // The gradient of the penalised objective is
  //     (dq - desired) + penalty * A' max(0, A dq - b),
  // whose Lipschitz constant is 1 + penalty * sigma_max(A)^2. Computing the
  // largest singular value exactly is cheap at this size and removes the
  // guesswork from the step size -- an overestimate wastes iterations, an
  // underestimate diverges.
  const Eigen::MatrixXd block = soft_matrix_.topRows(active_soft_rows_);
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(block);
  const double largest =
      svd.singularValues().size() > 0 ? svd.singularValues()(0) : 0.0;

  lipschitz_ = 1.0 + options_.penalty * largest * largest;
  lipschitz_dirty_ = false;
}

FilterResult SafetyFilter::solve(const JointVector& desired) {
  refresh_lipschitz();

  FilterResult result;

  // Clipping first means the starting iterate is already executable, so an
  // early return at any point below is still a valid command.
  JointVector current = clip(desired);

  if (active_soft_rows_ == 0) {
    result.dq = current;
    result.correction = (current - desired).norm();
    result.converged = true;
    return result;
  }

  const double step = 1.0 / lipschitz_;
  const Eigen::MatrixXd matrix = soft_matrix_.topRows(active_soft_rows_);
  const Eigen::VectorXd bound = soft_bound_.head(active_soft_rows_);

  JointVector momentum = current;
  double theta = 1.0;

  int iteration = 0;
  bool converged = false;

  for (iteration = 1; iteration <= options_.max_iterations; ++iteration) {
    const Eigen::VectorXd overshoot = (matrix * momentum - bound).cwiseMax(0.0);

    const JointVector gradient =
        (momentum - desired) + options_.penalty * (matrix.transpose() * overshoot);

    // The clip is the projection onto the box. Every iterate stays inside it,
    // which is why stopping early never produces an unexecutable command.
    const JointVector updated = clip(momentum - step * gradient);

    const double theta_next = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * theta * theta));
    momentum = updated + ((theta - 1.0) / theta_next) * (updated - current);

    const double change = (updated - current).cwiseAbs().maxCoeff();
    current = updated;
    theta = theta_next;

    if (change < options_.tolerance) {
      converged = true;
      break;
    }
  }

  const Eigen::VectorXd slack = (matrix * current - bound).cwiseMax(0.0);

  result.dq = current;
  result.correction = (current - desired).norm();
  result.relaxed = slack.size() > 0 ? slack.maxCoeff() : 0.0;
  result.iterations = iteration;
  result.converged = converged;
  return result;
}

}  // namespace stretchwbc
