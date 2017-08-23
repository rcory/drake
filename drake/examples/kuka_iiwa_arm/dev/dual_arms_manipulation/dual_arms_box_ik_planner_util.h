#pragma once

#include "drake/multibody/rigid_body_tree.h"
#include "drake/multibody/rigid_body_constraint.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
// Create a posture constraint, such that joints on some robots (either kuka1,
// kuka2 or the box) are fixed to the given posture `q`.
PostureConstraint FixRobotJoints(RigidBodyTreed* tree, const Eigen::VectorXd& q,
                                 bool fix_kuka1, bool fix_kuka2, bool fix_box);


Eigen::VectorXd GrabbingBoxFromTwoSides(RigidBodyTreed* tree, const Eigen::VectorXd& q, double box_size);
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
