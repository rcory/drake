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

// Computes the posture for the two kuka iiwa arms to grab the box from two sides.
// @param q. The current generalized position of the whole tree, including two
// kuka arms, and the box. The order is [q_left_kuka, q_right_kuka, q_box]
Eigen::VectorXd GrabbingBoxFromTwoSides(RigidBodyTreed* tree,
                                        const Eigen::VectorXd& q,
                                        double box_size);

// Computes the posture for the two kuka iiwa arms to move the box.
// @param q. The current generalized position of the whole tree.
// @param T_offset. The transform of the desired box pose in the world frame.
// @param kuka_box_fixed_link_indices The indices of links on kuka arms, that
// should be fixed in the box frame during the move.
Eigen::VectorXd MoveBox(RigidBodyTreed* tree, const Eigen::VectorXd& q,
                        const Eigen::Isometry3d& T_WB,
                        const std::vector<int>& kuka_box_fixed_link_indices);
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
