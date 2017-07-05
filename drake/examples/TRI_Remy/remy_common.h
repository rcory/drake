#pragma once
/// @file
///
/// This file assumes a 13-degree of freedom TRI-Remy mobile robot. Joints are
/// numbered sequentially starting from the base with the joint index
/// descriptions specified below.
/// Note: The four element base quaternion vector specifies the rotation of the
/// base and must be set accordingly. The individual quaternion vector elements
/// do not correspond to individual joints.
/// 0: base x-translation
/// 1: base y-translation
/// 2: base z-translation
/// 3: base quaternion element w
/// 4: base quaternion element x
/// 5: base quaternion element y
/// 6: base quaternion element z
/// 7: right_wheel
/// 8: left_wheel
/// 9: lift
/// 10: front_caster
/// 11: front_caster_wheel
/// 12: back_caster
/// 13: back_caster_wheel
/// --Jaco starts here--
/// 14: shoulder roll
/// 15: shoulder fore/aft
/// 16: elbow fore/aft
/// 17: forearm roll
/// 18: wrist yaw
/// 19: wrist roll
/// 20: finger 1 bend/extend
/// 21: finger 2 bend/extend
/// 22: finger 3 bend/extend
///
/// Rotational position/velocity units are in rad and rad/s, respectively.
/// Linear position units are in meters and m/s, respectively.

#include <string>

#include "drake/multibody/rigid_body_tree.h"

namespace drake {
namespace examples {
namespace Remy {

constexpr int kNumDofs = 22;  // DOFs available for the Remy robot

/// Verifies that @p tree matches assumptions about joint indices.
/// Aborts if the tree isn't as expected.
void VerifyRemyTree(const RigidBodyTree<double>& tree);

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
