#pragma once
/// @file
///
/// This file assumes a 27-degree of freedom TRI-Remy mobile robot. Joints are
/// numbered sequentially starting from the base with the joint index
/// descriptions specified below.
/// Notes:
/// - Naming matches output of RigidBodyTre<T>::computerPositionNameToIndexMap()
/// - The four element quaternion vectors specify rotation
///   and must be set accordingly. The individual quaternion vector elements
///   do not correspond to individual joints.
///
/// For POSITION index mapping:
/// 0: base_x
/// 1: base y
/// 2: base_z
/// 3: base_qw
/// 4: base_qx
/// 5: base_qy
/// 6: base_qz
/// 7: right_wheel_joint
/// 8: left_wheel_joint
/// 9: lift_joint
/// 10: front_ball_caster_joint_qw
/// 11: front_ball_caster_joint_qx
/// 12: front_ball_caster_joint_qy
/// 13: front_ball_caster_joint_qz
/// 14: back_ball_caster_joint_qw
/// 15: back_ball_caster_joint_qx
/// 16: back_ball_caster_joint_qy
/// 17: back_ball_caster_joint_qz
/// --Jaco starts here--
/// 18: j2n6s300_joint_1 (shoulder roll)
/// 19: j2n6s300_joint_2 (shoulder fore/aft)
/// 20: j2n6s300_joint_3 (elbow fore/aft)
/// 21: j2n6s300_joint_4 (forearm roll)
/// 22: j2n6s300_joint_5 (wrist yaw)
/// 23: j2n6s300_joint_6 (wrist roll)
/// 24: j2n6s300_joint_7 (finger 1 bend/extend)
/// 25: j2n6s300_joint_8 (finger 2 bend/extend)
/// 26: j2n6s300_joint_9 (finger 3 bend/extend)
///
/// For VELOCITY (offset) index mapping:
/// 0: base_x
/// 1: base_y
/// 2: base_z
/// 3: base_omega_x
/// 4: base_omega_y
/// 5: base_omega_z
/// 6: right_wheel_joint
/// 7: left_wheel_joint
/// 8: lift_joint
/// 9: front_ball_caster_omega_x
/// 10: front_ball_caster_omega_y
/// 11: front_ball_caster_omega_z
/// 12: back_ball_caster_omega_x
/// 13: back_ball_caster_omega_y
/// 14: back_ball_caster_omega_z
/// --Jaco starts here--
/// 15: j2n6s300_joint_1 (shoulder roll)
/// 16: j2n6s300_joint_2 (shoulder fore/aft)
/// 17: j2n6s300_joint_3 (elbow fore/aft)
/// 18: j2n6s300_joint_4 (forearm roll)
/// 19: j2n6s300_joint_5 (wrist yaw)
/// 20: j2n6s300_joint_6 (wrist roll)
/// 21: j2n6s300_joint_7 (finger 1 bend/extend)
/// 22: j2n6s300_joint_8 (finger 2 bend/extend)
/// 23: j2n6s300_joint_9 (finger 3 bend/extend)
///
/// Rotational position/velocity units are in rad and rad/s, respectively.
/// Linear position units are in meters and m/s, respectively.

#include <string>

#include "drake/multibody/rigid_body_tree.h"

namespace drake {
namespace examples {
namespace Remy {

// start indices for joint groups of interest
constexpr int kWheelQStart = 7;
constexpr int kWheelVStart = 6;

constexpr int kLiftQStart = 9;
constexpr int kLiftVStart = 8;

constexpr int kArmQStart = 18;
constexpr int kArmVStart = 15;

constexpr int kHandQStart = 24;
constexpr int kHandVStart = 21;

/// Verifies that @p tree matches assumptions about joint indices.
/// Aborts if the tree isn't as expected.
void VerifyRemyTree(const RigidBodyTree<double>& tree);

/// Prints out the @p tree
void PrintOutRemyTree(const RigidBodyTree<double>& tree);

/// Builds a RigidBodyTree at the specified @position and @orientation from
/// the model specified by @model_file_name.
/// This method is a convenience wrapper over `AddModelInstanceFromUrdfFile`.
/// @see drake::parsers::urdf::AddModelInstanceFromUrdfFile
void CreateTreeFromFloatingModelAtPose(
    const std::string& model_file_name, RigidBodyTreed* tree,
    const Eigen::Isometry3d& pose = Eigen::Isometry3d::Identity());

}  // namespace Remy
}  // namespace examples
}  // namespace drake
