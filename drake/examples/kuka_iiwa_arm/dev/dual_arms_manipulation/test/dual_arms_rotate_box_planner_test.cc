#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_rotate_box_planner.h"

#include <gtest/gtest.h>

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
GTEST_TEST(TestDualArmPlanner, TestCentroidalDynamicsConstraint) {
  CentroidalDynamicsConstraint cnstr(2, 0.01 * Eigen::Matrix3d::Identity(), IntegrationType::kBackwardEuler);
  AutoDiffVecXd x(39);
  for (int i = 0; i < 39; ++i) {
    x(i).value() = Eigen::Matrix<double, 1, 1>::Random()(0);
    x(i).derivatives() = Eigen::Matrix<double, 39, 1>::Zero();
    x(i).derivatives()(i) = 1.0;
  }
  AutoDiffVecXd y(6);
  cnstr.Eval(x, y);
}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake