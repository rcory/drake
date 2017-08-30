#pragma once

#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_box_util.h"
#include "drake/multibody/rigid_body_constraint.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
// Create a posture constraint, such that joints on some robots (either kuka1,
// kuka2 or the box) are fixed to the given posture `q`.
PostureConstraint FixRobotJoints(RigidBodyTreed* tree, const Eigen::VectorXd& q,
                                 bool fix_kuka1, bool fix_kuka2, bool fix_box);

class DualArmsBoxRotationPlanner {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DualArmsBoxRotationPlanner)

  DualArmsBoxRotationPlanner(RotateBox box_type, const Eigen::Isometry3d& right_kuka_base_pose);

  // Given the initial posture of the robot and the box, compute the final posture
  // that the robot can grab the box.
  Eigen::VectorXd GrabbingBoxFromTwoSides(const Eigen::VectorXd& q, double box_size) const;

  // Computes the posture for the two kuka iiwa arms to move the box.
  // @param q. The current generalized position of the whole tree.
  // @param T_offset. The transform of the desired box pose in the world frame.
  // @param kuka_box_fixed_link_indices The indices of links on kuka arms, that
  // should be fixed in the box frame during the move.
  Eigen::VectorXd MoveBox(const Eigen::VectorXd& q,
                          const Eigen::Isometry3d& T_WB,
                          const std::vector<int>& kuka_box_fixed_link_indices) const;

  RigidBodyTreed* tree() const {return tree_.get();}

  const std::array<int, 8>& left_iiwa_link_idx() const {return left_iiwa_link_idx_;}

  const std::array<int, 8>& right_iiwa_link_idx() const {return right_iiwa_link_idx_;}

 private:
  std::unique_ptr<RigidBodyTreed> tree_;
  int box_idx_;
  int l_hand_idx_;
  int r_hand_idx_;
  std::array<int, 8> left_iiwa_link_idx_;
  std::array<int, 8> right_iiwa_link_idx_;

  std::array<Eigen::Matrix3Xd, 8> iiwa_link_boundary_pts_;

  Eigen::Matrix<double, 3, 6> box_normals_;
};
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
