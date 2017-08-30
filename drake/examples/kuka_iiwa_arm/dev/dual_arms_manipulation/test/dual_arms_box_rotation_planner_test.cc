#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_box_rotation_planner.h"

#include "drake/lcm/drake_lcm.h"
#include "drake/manipulation/util/simple_tree_visualizer.h"
#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_box_util.h"
namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace {
int DoMain() {
  drake::lcm::DrakeLcm lcm;
  Eigen::Isometry3d right_kuka_base_offset;
  right_kuka_base_offset.setIdentity();
  right_kuka_base_offset.translation() = Eigen::Vector3d(-1, 0.5, 1);
  DualArmsBoxRotationPlanner planner(RotateBox::AmazonRubber, right_kuka_base_offset);
  Eigen::VectorXd q0 = Eigen::VectorXd::Zero(20);

  // zero configuration is a bad initial guess for Kuka.
  Eigen::Matrix<double, 7, 1> q_kuka0;
  q_kuka0 << 0, 0.5, 0.3, 0.3, 0.4, 0.5, 0.6;
  q0.topRows<7>() = q_kuka0;
  q0.middleRows<7>(7) = q_kuka0;
  Eigen::Vector3d box_pos(-0.5, 0.8, 1.3);
  Eigen::Vector3d box_rpy(0, 0, 0.3);
  q0.middleRows<3>(14) = box_pos;
  q0.bottomRows<3>() = box_rpy;

  Eigen::VectorXd q_sol = planner.GrabbingBoxFromTwoSides(q0, 0.5);
  manipulation::SimpleTreeVisualizer visualizer(*(planner.tree()), &lcm);
  visualizer.visualize(q_sol);


  KinematicsCache<double> cache = planner.tree()->CreateKinematicsCache();
  cache.initialize(q_sol);
  planner.tree()->doKinematics(cache);
  Eigen::Isometry3d base_pose = planner.tree()->relativeTransform(cache, 0, 1);
  std::cout << planner.tree()->get_body(1).get_name() << std::endl;
  std::cout << base_pose.matrix() << std::endl;


  Eigen::Isometry3d box_pose_des;
  box_pose_des.setIdentity();
  box_pose_des.translation() = right_kuka_base_offset.translation() + Eigen::Vector3d(0.5, 0.4, 0.4);
  q_sol = planner.MoveBox(q_sol, box_pose_des, {planner.left_iiwa_link_idx()[6], planner.right_iiwa_link_idx()[6]});
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
