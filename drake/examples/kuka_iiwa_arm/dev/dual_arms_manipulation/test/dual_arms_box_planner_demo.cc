#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_box_rotation_planner.h"

#include <lcm/lcm-cpp.hpp>

#include "external/lcmtypes_robotlocomotion/lcmtypes/robotlocomotion/robot_plan_t.hpp"
#include "optitrack/optitrack_frame_t.hpp"
#include "drake/common/find_resource.h"
#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_box_util.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_common.h"
#include "drake/examples/kuka_iiwa_arm/pick_and_place/pick_and_place_state_machine.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/manipulation/util/simple_tree_visualizer.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/util/lcmUtil.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace {
class MessageHandler : public lcm::DrakeLcmMessageHandlerInterface {
 public:
  MessageHandler() : received_msg_{} {}

  void HandleMessage(const std::string& channel, const void* message_buffer,
                     int message_size) override {
    std::lock_guard<std::mutex> lock(message_mutex_);
    received_msg_.decode(message_buffer, 0, message_size);
    received_channel_ = channel;
  }

  Eigen::Isometry3d GetBoxPose() const {
    Eigen::Isometry3d box_pose;
    box_pose.setIdentity();
    for (int i = 0; i < received_msg_.num_rigid_bodies; ++i) {
      if (received_msg_.rigid_bodies[i].id == 2) {
        // The box ID is 2.
        box_pose.linear() =
            Eigen::Quaterniond(received_msg_.rigid_bodies[i].quat[3],
                               received_msg_.rigid_bodies[i].quat[0],
                               received_msg_.rigid_bodies[i].quat[1],
                               received_msg_.rigid_bodies[i].quat[2])
                .toRotationMatrix();
        box_pose.translation() =
            Eigen::Vector3d(received_msg_.rigid_bodies[i].xyz[0],
                            received_msg_.rigid_bodies[i].xyz[1],
                            received_msg_.rigid_bodies[i].xyz[2]);
      }
    }
    return box_pose;
  }

  Eigen::Isometry3d GetRightKukaPose() const {
    Eigen::Isometry3d right_kuka_pose;
    right_kuka_pose.setIdentity();
    for (int i = 0; i < received_msg_.num_rigid_bodies; ++i) {
      if (received_msg_.rigid_bodies[i].id == 3) {
        right_kuka_pose.translation() =
            Eigen::Vector3d(received_msg_.rigid_bodies[i].xyz[0],
                            received_msg_.rigid_bodies[i].xyz[1],
                            received_msg_.rigid_bodies[i].xyz[2] - 0.03);
      }
    }
    return right_kuka_pose;
  }

  std::string get_received_channel() const { return received_channel_; }

 private:
  std::mutex message_mutex_;
  optitrack::optitrack_frame_t received_msg_;
  std::string received_channel_;
};

int DoMain() {
  drake::lcm::DrakeLcm drake_lcm;

  MessageHandler handler;
  drake_lcm.Subscribe("OPTITRACK_FRAMES", &handler);
  drake_lcm.StartReceiveThread();

  Eigen::Isometry3d box_pose;
  Eigen::Isometry3d right_kuka_pose;
  std::cout << "listening to OPTITRACK_FRAMES" << std::endl;
  while (true) {
    if (handler.get_received_channel() == "OPTITRACK_FRAMES") {
      box_pose = handler.GetBoxPose();
      right_kuka_pose = handler.GetRightKukaPose();
      drake_lcm.StopReceiveThread();
      break;
    }
  }
  std::cout << "Get optitrack_frame_t message from LCM" << std::endl;
  std::cout << "box_pose:\n" << box_pose.matrix() << std::endl;
  std::cout << "right_kuka_pose:\n" << right_kuka_pose.matrix() << std::endl;

  ::lcm::LCM lcm;

  pick_and_place::PickAndPlaceStateMachine::IiwaPublishCallback iiwa_callback =
      ([&](const robotlocomotion::robot_plan_t* plan) {
        lcm.publish("CANDIDATE_MANIP_PLAN", plan);
      });

  DualArmsBoxRotationPlanner planner(RotateBox::AmazonRubber, right_kuka_pose);
  auto tree = planner.tree();
  Eigen::VectorXd q0 = Eigen::VectorXd::Zero(20);

  // zero configuration is a bad initial guess for Kuka.
  Eigen::Matrix<double, 7, 1> q_kuka0;
  q_kuka0 << 0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01;
  q0.topRows<7>() = q_kuka0;
  q0.middleRows<7>(7) = q_kuka0;
  Eigen::Vector3d box_pos = box_pose.translation();
  Eigen::Vector3d box_rpy = math::rotmat2rpy(box_pose.linear());
  q0.middleRows<3>(14) = box_pos;
  q0.bottomRows<3>() = box_rpy;

  // Pre-grasping
  std::cout << "Pre-grasping\n";
  Eigen::VectorXd q1 = planner.GrabbingBoxFromTwoSides(q0, 0.7);
  // Grasping
  std::cout << "Grasping\n";
  Eigen::VectorXd q2 = planner.GrabbingBoxFromTwoSides(q1, 0.46);

  while (true) {
    if (handler.get_received_channel() == "OPTITRACK_FRAMES") {
      box_pose = handler.GetBoxPose();
      right_kuka_pose = handler.GetRightKukaPose();
      drake_lcm.StopReceiveThread();
      break;
    }
  }
  // Lift up the box
  Eigen::Matrix3d box_normal_facing_world_xyz =
      planner.BoxNormalFacingWorldXYZ(box_pose);
  std::cout << "Lift up the box.\n";
  Eigen::Isometry3d box_up_pose;
  box_up_pose.setIdentity();
  box_up_pose.linear() = box_normal_facing_world_xyz.transpose();
  box_up_pose.translation() =
      right_kuka_pose.translation() + Eigen::Vector3d(0.5, 0.5, 0.4);
  Eigen::VectorXd q3 = planner.MoveBox(
      q2, box_up_pose,
      {planner.left_iiwa_link_idx()[6], planner.right_iiwa_link_idx()[6]});

  // Put down the box
  std::cout << "Put down the box.\n";
  Eigen::Isometry3d box_down_pose;
  box_down_pose.setIdentity();
  box_down_pose.linear() = box_normal_facing_world_xyz.transpose();
  box_down_pose.translation() =
      right_kuka_pose.translation() + Eigen::Vector3d(0.5, 0.5, 0.27);
  Eigen::VectorXd q4 = planner.MoveBox(
      q3, box_down_pose,
      {planner.left_iiwa_link_idx()[6], planner.right_iiwa_link_idx()[6]});

  // Release the box
  std::cout << "Release the box.\n";
  Eigen::VectorXd q5 = planner.GrabbingBoxFromTwoSides(q4, 0.6);

  Eigen::MatrixXd keyframes(14, 7);
  keyframes.col(0) = q0.topRows<14>();
  keyframes.col(1) = q0.topRows<14>();
  keyframes.col(2) = q1.topRows<14>();
  keyframes.col(3) = q2.topRows<14>();
  keyframes.col(4) = q3.topRows<14>();
  keyframes.col(5) = q4.topRows<14>();
  keyframes.col(6) = q5.topRows<14>();

  std::vector<double> times{0, 2, 4, 4.5, 5.5, 6.5, 7.5};

  robotlocomotion::robot_plan_t plan{};

  auto iiwa_tree = std::make_unique<RigidBodyTree<double>>();
//  const std::string iiwa_path = FindResourceOrThrow(
//      "drake/manipulation/models/iiwa_description/urdf/"
//      "dual_iiwa14_polytope_collision.urdf");
  const std::string iiwa_path = FindResourceOrThrow(
      "drake/manipulation/models/iiwa_description/urdf/"
          "dual_iiwa14_primitive_sphere_visual_collision.urdf");
  parsers::urdf::AddModelInstanceFromUrdfFileToWorld(
      iiwa_path, multibody::joints::kFixed, iiwa_tree.get());
  const RigidBodyTree<double>& iiwa = *(iiwa_tree.get());

  std::vector<int> info(times.size(), 1);
  *(&plan) = EncodeKeyFrames(iiwa, times, info, keyframes);
  iiwa_callback(&plan);

  manipulation::SimpleTreeVisualizer visualizer(*tree, &drake_lcm);
  visualizer.visualize(q3);
  KinematicsCache<double> cache = tree->CreateKinematicsCache();
  cache.initialize(q2);
  tree->doKinematics(cache);
  Eigen::Isometry3d base_pose = tree->relativeTransform(cache, 0, 1);
  std::cout << tree->get_body(1).get_name() << std::endl;
  std::cout << base_pose.matrix() << std::endl;
  return 0;
}
}  // namespace
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main() { return drake::examples::kuka_iiwa_arm::DoMain(); }
