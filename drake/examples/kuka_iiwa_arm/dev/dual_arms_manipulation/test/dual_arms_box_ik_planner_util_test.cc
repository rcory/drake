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
  auto tree = ConstructDualArmAndBox();
  manipulation::SimpleTreeVisualizer visualizer(*tree, &lcm);
  Eigen::VectorXd q0 = Eigen::VectorXd::Zero(20);
  visualizer.visualize(q0);
  return 0;
}
}  // namespace
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main() {
  return drake::examples::kuka_iiwa_arm::DoMain();
}
