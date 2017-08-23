#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_box_ik_planner_util.h"

#include "drake/lcm/drake_lcm.h"
#include "drake/manipulation/util/simple_tree_visualizer.h"
#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_box_util.h"
namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace {
int DoMain() {
  drake::lcm::DrakeLcm lcm;
  auto tree = ConstructDualArmAndBox(RotateBox::AmazonRubber);
  Eigen::VectorXd q0 = Eigen::VectorXd::Zero(20);

  // zero configuration is a bad initial guess for Kuka.
  Eigen::Matrix<double, 7, 1> q_kuka0;
  q_kuka0 << 0, 0.5, 0.3, 0.3, 0.4, 0.5, 0.6;
  q0.topRows<7>() = q_kuka0;
  q0.middleRows<7>(7) = q_kuka0;
  Eigen::Vector3d box_pos(0.5, 0.45, 0.3);
  Eigen::Vector3d box_rpy(0, 0, 0.3);
  q0.middleRows<3>(14) = box_pos;
  q0.bottomRows<3>() = box_rpy;

  Eigen::VectorXd q_sol = GrabbingBoxFromTwoSides(tree.get(), q0, 0.5);
  manipulation::SimpleTreeVisualizer visualizer(*tree, &lcm);
  visualizer.visualize(q_sol);
  return 0;
}
}  // namespace
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main() {
  return drake::examples::kuka_iiwa_arm::DoMain();
}
