#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_rotate_box_planner.h"

#include "drake/common/drake_path.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/parsers/sdf_parser.h"
#include "drake/multibody/joints/fixed_joint.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
std::unique_ptr<RigidBodyTreed> ConstructDualArmAndBox() {
  std::unique_ptr<RigidBodyTree<double>> rigid_body_tree =
      std::make_unique<RigidBodyTree<double>>();
  const std::string model_path = drake::GetDrakePath() +
      "/manipulation/models/iiwa_description/urdf/"
          "iiwa14_polytope_collision.urdf";

  const std::string table_path =
      drake::GetDrakePath() + "/examples/kuka_iiwa_arm/models/table/"
          "extra_heavy_duty_table_surface_only_collision.sdf";

  auto table1_frame = std::make_shared<RigidBodyFrame<double>>(
      "iiwa_table1",
      rigid_body_tree->get_mutable_body(0),
      Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero());

  auto table2_frame = std::make_shared<RigidBodyFrame<double>>(
      "iiwa_table2",
      rigid_body_tree->get_mutable_body(0),
      Eigen::Vector3d(0.8, 0, 0),
      Eigen::Vector3d::Zero());

  parsers::sdf::AddModelInstancesFromSdfFile(table_path,
                                             drake::multibody::joints::kFixed,
                                             table1_frame,
                                             rigid_body_tree.get());

  parsers::sdf::AddModelInstancesFromSdfFile(table_path,
                                             drake::multibody::joints::kFixed,
                                             table2_frame,
                                             rigid_body_tree.get());

  const double kTableTopZInWorld = 0.736 + 0.057 / 2;
  const Eigen::Vector3d kRobotBase1(-0.243716, -0.625087, kTableTopZInWorld);
  const Eigen::Vector3d kRobotBase2(kRobotBase1(0) + 0.8, -0.625087, kTableTopZInWorld);

  auto robot_base_frame1 = std::make_shared<RigidBodyFrame<double>>(
      "iiwa_base1",
      rigid_body_tree->get_mutable_body(0),
      kRobotBase1,
      Eigen::Vector3d::Zero());
  rigid_body_tree->addFrame(robot_base_frame1);

  parsers::urdf::AddModelInstanceFromUrdfFile(
      model_path,
      drake::multibody::joints::kFixed,
      robot_base_frame1,
      rigid_body_tree.get());

  auto robot_base_frame2 = std::make_shared<RigidBodyFrame<double>>(
      "iiwa_base2",
      rigid_body_tree->get_mutable_body(0),
      kRobotBase2,
      Eigen::Vector3d::Zero());
  rigid_body_tree->addFrame(robot_base_frame2);

  parsers::urdf::AddModelInstanceFromUrdfFile(
      model_path,
      drake::multibody::joints::kFixed,
      robot_base_frame2,
      rigid_body_tree.get());

  return rigid_body_tree;
}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake