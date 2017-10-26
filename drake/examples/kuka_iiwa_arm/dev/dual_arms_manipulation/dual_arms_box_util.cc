#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_box_util.h"

#include "drake/common/find_resource.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/lcmtypes/drake/lcmt_viewer_load_robot.hpp"
#include "drake/multibody/joints/fixed_joint.h"
#include "drake/multibody/parsers/sdf_parser.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_plant/create_load_robot_message.h"
#include "drake/multibody/rigid_body_plant/viewer_draw_translator.h"
#include "drake/multibody/shapes/geometry.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
void AddSphereToBody(RigidBodyTreed* tree, int link_idx,
                     const Eigen::Vector3d& pt, const std::string& name,
                     double radius) {
  auto sphere_body = std::make_unique<RigidBody<double>>();
  sphere_body->set_name(name);
  sphere_body->set_mass(0.001);
  sphere_body->set_spatial_inertia(Matrix6<double>::Identity() * 1E-8);

  const DrakeShapes::Sphere shape(radius);
  const Eigen::Vector4d material(1, 0, 0, 1);

  const DrakeShapes::VisualElement visual_element(
      shape, Eigen::Isometry3d::Identity(), material);

  sphere_body->AddVisualElement(visual_element);

  Eigen::Isometry3d joint_transform;
  joint_transform.linear() = Eigen::Matrix3d::Identity();
  joint_transform.translation() = pt;

  auto joint = std::make_unique<FixedJoint>(name + "_joint", joint_transform);
  auto link = tree->get_mutable_body(link_idx);
  sphere_body->add_joint(link, std::move(joint));
  tree->bodies.push_back(std::move(sphere_body));
}

std::unique_ptr<RigidBodyTreed> ConstructDualArmAndBox() {
  std::unique_ptr<RigidBodyTree<double>> rigid_body_tree =
      std::make_unique<RigidBodyTree<double>>();
  const std::string model_path = FindResourceOrThrow(
      "drake/manipulation/models/iiwa_description/urdf/"
      "dual_iiwa14_polytope_collision.urdf");

  parsers::urdf::AddModelInstanceFromUrdfFile(model_path,
                                              drake::multibody::joints::kFixed,
                                              nullptr, rigid_body_tree.get());

  const std::string box_path = FindResourceOrThrow(
      "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/box.urdf");

  parsers::urdf::AddModelInstanceFromUrdfFile(
      box_path, drake::multibody::joints::kRollPitchYaw, nullptr,
      rigid_body_tree.get());

  // AddSphereToBody(rigid_body_tree.get(),
  // rigid_body_tree->FindBodyIndex("left_iiwa_link_4"), Eigen::Vector3d(0, 0,
  // 0), "left_iiwa_link_4_pt0", 0.01);
  // AddSphereToBody(rigid_body_tree.get(),
  // rigid_body_tree->FindBodyIndex("left_iiwa_link_4"), Eigen::Vector3d(0, 0,
  // 0.1), "left_iiwa_link_4_pt1", 0.01);
  // AddSphereToBody(rigid_body_tree.get(),
  // rigid_body_tree->FindBodyIndex("left_iiwa_link_4"), Eigen::Vector3d(0,
  // 0.08, 0.1), "left_iiwa_link_4_pt2", 0.01);
  // AddSphereToBody(rigid_body_tree.get(),
  // rigid_body_tree->FindBodyIndex("left_iiwa_link_4"), Eigen::Vector3d(0,
  // 0.05, 0), "left_iiwa_link_4_pt3", 0.01);
  AddSphereToBody(rigid_body_tree.get(),
                  rigid_body_tree->FindBodyIndex("left_iiwa_link_ee_kuka"),
                  Eigen::Vector3d(0, 0, 0.03), "left_iiwa_link_ee_kuka_pt1",
                  0.01);
  AddSphereToBody(rigid_body_tree.get(), rigid_body_tree->FindBodyIndex("box"),
                  Eigen::Vector3d(0.28, 0, 0), "box_+x_center", 0.01);
  AddSphereToBody(rigid_body_tree.get(), rigid_body_tree->FindBodyIndex("box"),
                  Eigen::Vector3d(-0.28, 0, 0), "box_-x_center", 0.01);
  AddSphereToBody(rigid_body_tree.get(), rigid_body_tree->FindBodyIndex("box"),
                  Eigen::Vector3d(0, 0.28, 0), "box_+y_center", 0.01);
  AddSphereToBody(rigid_body_tree.get(), rigid_body_tree->FindBodyIndex("box"),
                  Eigen::Vector3d(0, -0.28, 0), "box_-y_center", 0.01);
  return rigid_body_tree;
}

void VisualizePosture(RigidBodyTreed* tree,
                      const Eigen::Ref<const Eigen::VectorXd>& q_kuka1,
                      const Eigen::Ref<const Eigen::VectorXd>& q_kuka2,
                      const Eigen::Ref<const Eigen::VectorXd>& q_box) {
  lcm::DrakeLcm lcm;
  std::vector<uint8_t> message_bytes;

  lcmt_viewer_load_robot load_msg =
      multibody::CreateLoadRobotMessage<double>(*tree);

  const int length = load_msg.getEncodedSize();
  message_bytes.resize(length);
  load_msg.encode(message_bytes.data(), 0, length);
  lcm.Publish("DRAKE_VIEWER_LOAD_ROBOT", message_bytes.data(),
              message_bytes.size());

  systems::ViewerDrawTranslator posture_drawer(*tree);
  Eigen::VectorXd x(tree->get_num_positions() + tree->get_num_velocities());
  x.block<7, 1>(tree->FindBody("left_iiwa_link_0")->get_position_start_index(),
                0) = q_kuka1;
  x.block<7, 1>(tree->FindBody("right_iiwa_link_0")->get_position_start_index(),
                0) = q_kuka2;
  x.block<6, 1>(tree->FindBody("box")->get_position_start_index(), 0) = q_box;
  systems::BasicVector<double> q_draw(x);
  posture_drawer.Serialize(0, q_draw, &message_bytes);
  lcm.Publish("DRAKE_VIEWER_DRAW", message_bytes.data(), message_bytes.size());
}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake