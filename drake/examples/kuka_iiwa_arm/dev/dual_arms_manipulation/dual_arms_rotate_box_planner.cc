#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_rotate_box_planner.h"

#include "drake/common/drake_path.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/parsers/sdf_parser.h"
#include "drake/multibody/joints/fixed_joint.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/rigid_body_plant/viewer_draw_translator.h"
#include "drake/multibody/rigid_body_plant/create_load_robot_message.h"
#include "drake/lcmtypes/drake/lcmt_viewer_load_robot.hpp"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
std::unique_ptr<RigidBodyTreed> ConstructDualArmAndBox() {
  std::unique_ptr<RigidBodyTree<double>> rigid_body_tree =
      std::make_unique<RigidBodyTree<double>>();
  const std::string model_path = drake::GetDrakePath() +
      "/manipulation/models/iiwa_description/urdf/"
          "dual_iiwa14_polytope_collision.urdf";

  parsers::urdf::AddModelInstanceFromUrdfFile(model_path, drake::multibody::joints::kFixed, nullptr, rigid_body_tree.get());

  return rigid_body_tree;
}

void VisualizePosture(RigidBodyTreed* tree, const Eigen::Ref<const Eigen::VectorXd>& q_kuka1, const Eigen::Ref<const Eigen::VectorXd>& q_kuka2, const Eigen::Ref<Eigen::Matrix<double, 7, 1>>& q_box) {
  lcm::DrakeLcm lcm;
  std::vector<uint8_t> message_bytes;

  lcmt_viewer_load_robot  load_msg = multibody::CreateLoadRobotMessage<double>(*tree);

  const int length = load_msg.getEncodedSize();
  message_bytes.resize(length);
  load_msg.encode(message_bytes.data(), 0, length);
  lcm.Publish("DRAKE_VIEWER_LOAD_ROBOT", message_bytes.data(), message_bytes.size());

  systems::ViewerDrawTranslator posture_drawer(*tree);
  Eigen::VectorXd x(tree->get_num_positions() + tree->get_num_velocities());
  x.block<7, 1>(tree->FindBody("left_iiwa_link_0")->get_position_start_index(), 0) = q_kuka1;
  x.block<7, 1>(tree->FindBody("right_iiwa_link_0")->get_position_start_index(), 0) = q_kuka2;
  systems::BasicVector<double> q_draw(x);
  posture_drawer.Serialize(0, q_draw, &message_bytes);
  lcm.Publish("DRAKE_VIEWER_DRAW", message_bytes.data(),
              message_bytes.size());
}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake