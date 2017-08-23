#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_box_ik_planner_util.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
PostureConstraint FixRobotJoints(RigidBodyTreed* tree, const Eigen::VectorXd& q,
                                 bool fix_kuka1, bool fix_kuka2, bool fix_box) {
  PostureConstraint posture_cnstr(tree);
  Eigen::VectorXd q_lb = Eigen::Matrix<double, 20, 1>::Constant(
      -std::numeric_limits<double>::infinity());
  Eigen::VectorXd q_ub = Eigen::Matrix<double, 20, 1>::Constant(
      std::numeric_limits<double>::infinity());
  if (fix_kuka1) {
    for (int i = 0; i < 7; ++i) {
      q_lb(i) = q(i);
      q_ub(i) = q(i);
    }
  }
  if (fix_kuka2) {
    for (int i = 7; i < 14; ++i) {
      q_lb(i) = q(i);
      q_ub(i) = q(i);
    }
  }
  if (fix_box) {
    for (int i = 14; i < 20; ++i) {
      q_lb(i) = q(i);
      q_ub(i) = q(i);
    }
  }
  Eigen::VectorXi q_idx(20);
  for (int i = 0; i < 20; ++i) {
    q_idx(i) = i;
  }
  posture_cnstr.setJointLimits(q_idx, q_lb, q_ub);
  return posture_cnstr;
}

Eigen::VectorXd GrabbingBoxFromTwoSides(RigidBodyTreed* tree, const Eigen::VectorXd& q, double box_size) {
  KinematicsCache<double> cache = tree->CreateKinematicsCache();
  cache.initialize(q);
  tree->doKinematics(cache);
  int box_idx = tree->FindBodyIndex("box");
  const auto box_pose = tree->relativeTransform(cache, 0, box_idx);
  const Eigen::Matrix<double, 6, 1> q_box = q.bottomRows<6>();
  int l_hand_idx = tree->FindBodyIndex("left_iiwa_link_ee_kuka");
  int r_hand_idx = tree->FindBodyIndex("right_iiwa_link_ee_kuka");

  // First determine which face of the box should be grabbed by the left kuka.
  // The normal on this face should be approximately equal to [0;1;0]
  Eigen::Matrix<double, 3, 6> box_normals;
  box_normals << Eigen::Matrix3d::Identity(), -Eigen::Matrix3d::Identity();
  // box_pose.linear() * box_left_face_normal approximately equals to [0;1;0]
  Eigen::Matrix<double, 1, 6> left_face_normal_angle = Eigen::Vector3d(0, 1, 0).transpose() * box_normals;
  int box_left_face_normal_idx;
  left_face_normal_angle.maxCoeff(&box_left_face_normal_idx);
  const Eigen::Vector3d box_left_face_normal = box_normals.col(box_left_face_normal_idx);

  // Now determines which face of the box should be grabbed by the right kuka.
  // The normal on this face should be approximately equal to [0; -1; 0]
  Eigen::Matrix<double, 1, 6> right_face_normal_angle = Eigen::Vector3d(0, -1, 0).transpose() * box_normals;
  int box_right_face_normal_idx;
  right_face_normal_angle.maxCoeff(&box_right_face_normal_idx);
  const Eigen::Vector3d box_right_face_normal = box_normals.col(box_right_face_normal_idx);
  // Constrain the left hand to align with the box.

  // Now determines which face of the box faces the robot. The normal on this
  // face should be approximately equal to [1; 0; 0], if box_pose.translation(0) < 0.
  Eigen::Matrix<double, 1, 6> front_face_normal_angle;
  if (box_pose.translation()(0) < 0) {
    front_face_normal_angle = Eigen::Vector3d(1, 0, 0).transpose() * box_normals;
  } else {
    front_face_normal_angle = Eigen::Vector3d(-1, 0, 0).transpose() * box_normals;
  }
  int box_front_face_normal_idx;
  front_face_normal_angle.maxCoeff(&box_front_face_normal_idx);
  const Eigen::Vector3d box_front_face_normal = box_normals.col(box_front_face_normal);

  
  return q;
}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
