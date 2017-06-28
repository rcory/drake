#pragma once
/// @file
///
/// This file assumes a 10-degree of freedom TRI-Remy mobile robot. Joints are
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
/// 7: right wheel
/// 8: left wheel
/// 9: lift
///
/// Rotational position/velocity units are in rad and rad/s, respectively.
/// Linear position units are in meters and m/s, respectively.

#include <string>

#include "drake/multibody/rigid_body_tree.h"

namespace drake {
namespace examples {
namespace Remy {

constexpr int kNumDofs = 10;  // DOFs available for the Remy robot

constexpr int kBasexTranslationIdx = 0;
constexpr int kBaseyTranslationIdx = 1;
constexpr int kBasezTranslationIdx = 2;
constexpr int kQuatwElementIdx = 3;
constexpr int kQuatxElementIdx = 4;
constexpr int kQuatyElementIdx = 5;
constexpr int kQuatzElementIdx = 6;
constexpr int kRWheelJointIdx = 7;
constexpr int kLWheelJointIdx = 8;
constexpr int kLiftJointIdx = 9;


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
