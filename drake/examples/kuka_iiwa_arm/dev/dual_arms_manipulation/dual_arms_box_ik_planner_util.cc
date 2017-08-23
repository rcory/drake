#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_box_ik_planner_util.h"
#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_box_util.h"
#include "drake/multibody/rigid_body_ik.h"

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

Eigen::VectorXd GrabbingBoxFromTwoSides(RigidBodyTreed* tree,
                                        const Eigen::VectorXd& q,
                                        double box_size) {
  KinematicsCache<double> cache = tree->CreateKinematicsCache();
  cache.initialize(q);
  tree->doKinematics(cache);
  int box_idx = tree->FindBodyIndex("box");
  const auto box_pose = tree->relativeTransform(cache, 0, box_idx);
  std::cout << "box_pose:\n" << box_pose.matrix() << std::endl;
  int l_hand_idx = tree->FindBodyIndex("left_iiwa_link_ee_kuka");
  int r_hand_idx = tree->FindBodyIndex("right_iiwa_link_ee_kuka");
  int l_iiwa_link6_idx = tree->FindBodyIndex("left_iiwa_link_6");
  int l_iiwa_link5_idx = tree->FindBodyIndex("left_iiwa_link_5");
  int r_iiwa_link6_idx = tree->FindBodyIndex("right_iiwa_link_6");
  int r_iiwa_link5_idx = tree->FindBodyIndex("right_iiwa_link_5");
  int r_iiwa_link3_idx = tree->FindBodyIndex("right_iiwa_link_3");

  // First determine which face of the box should be grabbed by the left kuka.
  // The normal on this face should be approximately equal to [0;1;0]
  Eigen::Matrix<double, 3, 6> box_normals;
  box_normals << Eigen::Matrix3d::Identity(), -Eigen::Matrix3d::Identity();
  // box_pose.linear() * box_left_face_normal approximately equals to [0;1;0]
  Eigen::Matrix<double, 1, 6> left_face_normal_angle =
      Eigen::Vector3d(0, 1, 0).transpose() * box_pose.linear() * box_normals;
  int box_left_face_normal_idx;
  left_face_normal_angle.maxCoeff(&box_left_face_normal_idx);
  const Eigen::Vector3d box_left_face_normal =
      box_normals.col(box_left_face_normal_idx);

  // Now determines which face of the box should be grabbed by the right kuka.
  // The normal on this face should be approximately equal to [0; -1; 0]
  Eigen::Matrix<double, 1, 6> right_face_normal_angle =
      Eigen::Vector3d(0, -1, 0).transpose() * box_pose.linear() * box_normals;
  int box_right_face_normal_idx;
  right_face_normal_angle.maxCoeff(&box_right_face_normal_idx);
  const Eigen::Vector3d box_right_face_normal =
      box_normals.col(box_right_face_normal_idx);
  // Constrain the left hand to align with the box.

  // Now determines which face of the box faces the robot. The normal on this
  // face should be approximately equal to [1; 0; 0], if box_pose.translation(0)
  // < 0.
  Eigen::Matrix<double, 1, 6> front_face_normal_angle;
  if (box_pose.translation()(0) < 0) {
    front_face_normal_angle =
        Eigen::Vector3d(1, 0, 0).transpose() * box_pose.linear() * box_normals;
  } else {
    front_face_normal_angle =
        Eigen::Vector3d(-1, 0, 0).transpose() * box_pose.linear() * box_normals;
  }
  int box_front_face_normal_idx;
  front_face_normal_angle.maxCoeff(&box_front_face_normal_idx);
  const Eigen::Vector3d box_front_face_normal =
      box_normals.col(box_front_face_normal_idx);

  // Now determines which face of the box faces upward. The normal on that
  // face should be approximately equal to [0;0;1].
  Eigen::Matrix<double, 1, 6> top_face_normal_angle =
      Eigen::Vector3d(0, 0, 1).transpose() * box_pose.linear() * box_normals;
  int box_top_face_normal_idx;
  top_face_normal_angle.maxCoeff(&box_top_face_normal_idx);
  const Eigen::Vector3d box_top_face_normal =
      box_normals.col(box_top_face_normal_idx);

  // Solve an IK problem on both left and right kuka
  PostureConstraint box_fixed_pose =
      FixRobotJoints(tree, q, false, false, true);

  // left palm axis aligns with the box, pointing outward.
  WorldGazeDirConstraint left_palm_axis_cnstr(
      tree, l_hand_idx, Eigen::Vector3d(0, 0, 1),
      -box_pose.linear() * box_front_face_normal, 0);
  // right palm axis aligns with the box, pointing outward.
  WorldGazeDirConstraint right_palm_axis_cnstr(
      tree, r_hand_idx, Eigen::Vector3d(0, 0, 1),
      -box_pose.linear() * box_front_face_normal, 0);

  const double palm_radius = 0.06;
  // left palm contact region
  Eigen::Matrix<double, 3, 4> left_palm_contact_region =
      ((box_size / 2 + palm_radius) * box_left_face_normal) *
      Eigen::RowVector4d::Ones();
  left_palm_contact_region.col(0) += -box_size * 0.4 * box_top_face_normal;
  left_palm_contact_region.col(1) += box_size * 0.4 * box_top_face_normal;
  left_palm_contact_region.col(2) =
      left_palm_contact_region.col(0) - box_size * 0.45 * box_front_face_normal;
  left_palm_contact_region.col(3) =
      left_palm_contact_region.col(1) - box_size * 0.45 * box_front_face_normal;
  // AddSphereToBody(tree, box_idx, left_palm_contact_region.col(0),
  // "left_region_pt0", 0.01);
  // AddSphereToBody(tree, box_idx, left_palm_contact_region.col(1),
  // "left_region_pt1", 0.01);
  // AddSphereToBody(tree, box_idx, left_palm_contact_region.col(2),
  // "left_region_pt2", 0.01);
  // AddSphereToBody(tree, box_idx, left_palm_contact_region.col(3),
  // "left_region_pt3", 0.01);
  AddSphereToBody(tree, l_iiwa_link6_idx, Eigen::Vector3d::Zero(),
                  "l_iiwa_link6_pt0", 0.005);
  AddSphereToBody(tree, l_iiwa_link6_idx, Eigen::Vector3d(0, 0, -0.09),
                  "l_iiwa_link6_pt1", 0.005);
  AddSphereToBody(tree, l_iiwa_link6_idx, Eigen::Vector3d(0.04, 0, -0.09),
                  "l_iiwa_link6_pt2", 0.005);
  AddSphereToBody(tree, l_iiwa_link6_idx, Eigen::Vector3d(-0.04, 0, -0.09),
                  "l_iiwa_link6_pt3", 0.005);
  AddSphereToBody(tree, l_iiwa_link6_idx, Eigen::Vector3d(0, 0.04, -0.09),
                  "l_iiwa_link6_pt4", 0.005);
  AddSphereToBody(tree, l_iiwa_link6_idx, Eigen::Vector3d(0, -0.04, -0.09),
                  "l_iiwa_link6_pt5", 0.005);

  AddSphereToBody(tree, l_iiwa_link5_idx, Eigen::Vector3d::Zero(),
                  "l_iiwa_link5_pt0", 0.01);
  AddSphereToBody(tree, l_iiwa_link5_idx, Eigen::Vector3d(0.07, 0, 0),
                  "l_iiwa_link5_pt1", 0.005);
  AddSphereToBody(tree, l_iiwa_link5_idx, Eigen::Vector3d(0, 0.07, 0),
                  "l_iiwa_link5_pt2", 0.005);
  AddSphereToBody(tree, l_iiwa_link5_idx, Eigen::Vector3d(0, 0, 0.1),
                  "l_iiwa_link5_pt3", 0.005);
  AddSphereToBody(tree, r_iiwa_link6_idx, Eigen::Vector3d::Zero(),
                  "r_iiwa_link6_pt0", 0.01);
  AddSphereToBody(tree, r_iiwa_link5_idx, Eigen::Vector3d::Zero(),
                  "r_iiwa_link5_pt0", 0.01);

  Eigen::Vector3d left_palm_contact_region_lb =
      left_palm_contact_region.rowwise().minCoeff();
  Eigen::Vector3d left_palm_contact_region_ub =
      left_palm_contact_region.rowwise().maxCoeff();
  WorldPositionInFrameConstraint left_palm_contact_constraint(
      tree, l_hand_idx, Eigen::Vector3d::Zero(), box_pose.matrix(),
      left_palm_contact_region_lb, left_palm_contact_region_ub);

  // Add the non-penetration constraint between left iiwa link6 and the box
  Eigen::Matrix<double, 3, 4> l_iiwa_link6_pts;
  // clang-format off
  l_iiwa_link6_pts << 0.04, -0.04, 0, 0,
                      0, 0, 0.04, -0.04,
                      -0.09, -0.09, -0.09, -0.09;
  // clang-format on
  Eigen::Matrix<double, 3, 4> left_face_nonpenetration_lb =
      Eigen::Matrix<double, 3, 4>::Constant(NAN);
  Eigen::Matrix<double, 3, 4> left_face_nonpenetration_ub =
      Eigen::Matrix<double, 3, 4>::Constant(NAN);
  if (box_left_face_normal_idx < 3) {
    left_face_nonpenetration_lb.row(box_left_face_normal_idx) =
        Eigen::RowVector4d::Constant(box_size / 2);
  } else {
    left_face_nonpenetration_ub.row(box_left_face_normal_idx - 3) =
        Eigen::RowVector4d::Constant(-box_size / 2);
  }
  WorldPositionInFrameConstraint l_iiwa_link6_nonpenetration_cnstr(
      tree, l_iiwa_link6_idx, l_iiwa_link6_pts, box_pose.matrix(),
      left_face_nonpenetration_lb, left_face_nonpenetration_ub);

  // Add the non-penetration constraint between left iiwa link5 and the box
  Eigen::Matrix<double, 3, 4> l_iiwa_link5_pts;
  // clang-format off
  l_iiwa_link5_pts << 0.07, -0.07, 0, 0,
                      0, 0, 0.07, -0.07,
                      0, 0, 0, 0;
  // clang-format on
  WorldPositionInFrameConstraint l_iiwa_link5_nonpenetration_cnstr(
      tree, l_iiwa_link5_idx, l_iiwa_link5_pts, box_pose.matrix(),
      left_face_nonpenetration_lb, left_face_nonpenetration_ub);

  // right palm contact region
  Eigen::Matrix<double, 3, 4> right_palm_contact_region =
      ((box_size / 2 + palm_radius) * box_right_face_normal) *
      Eigen::RowVector4d::Ones();
  right_palm_contact_region.col(0) += -box_size * 0.4 * box_top_face_normal;
  right_palm_contact_region.col(1) += box_size  * 0.4 * box_top_face_normal;
  right_palm_contact_region.col(2) =
      right_palm_contact_region.col(0) - box_size * 0.45 * box_front_face_normal;
  right_palm_contact_region.col(3) =
      right_palm_contact_region.col(1) - box_size * 0.45 * box_front_face_normal;
  // AddSphereToBody(tree, box_idx, right_palm_contact_region.col(0),
  // "right_region_pt0", 0.01);
  // AddSphereToBody(tree, box_idx, right_palm_contact_region.col(1),
  // "right_region_pt1", 0.01);
  // AddSphereToBody(tree, box_idx, right_palm_contact_region.col(2),
  // "right_region_pt2", 0.01);
  // AddSphereToBody(tree, box_idx, right_palm_contact_region.col(3),
  // "right_region_pt3", 0.01);

  // The surface of the right palm should touch the box right face.
  Eigen::Vector3d right_palm_contact_region_lb =
      right_palm_contact_region.rowwise().minCoeff();
  Eigen::Vector3d right_palm_contact_region_ub =
      right_palm_contact_region.rowwise().maxCoeff();
  WorldPositionInFrameConstraint right_palm_contact_constraint(
      tree, r_hand_idx, Eigen::Vector3d::Zero(), box_pose.matrix(),
      right_palm_contact_region_lb, right_palm_contact_region_ub);

  // Add the non-penetration constraint on the right face
  Eigen::Matrix<double, 3, 4> right_face_nonpenetration_lb =
      Eigen::Matrix<double, 3, 4>::Constant(NAN);
  Eigen::Matrix<double, 3, 4> right_face_nonpenetration_ub =
      Eigen::Matrix<double, 3, 4>::Constant(NAN);
  if (box_right_face_normal_idx < 3) {
    right_face_nonpenetration_lb.row(box_right_face_normal_idx) =
        Eigen::RowVector4d::Constant(box_size / 2);
  } else {
    right_face_nonpenetration_ub.row(box_right_face_normal_idx - 3) =
        Eigen::RowVector4d::Constant(-box_size / 2);
  }
  Eigen::Matrix<double, 3, 4> r_iiwa_link5_pts = l_iiwa_link5_pts;
  WorldPositionInFrameConstraint r_iiwa_link5_nonpenetration_cnstr{tree, r_iiwa_link5_idx, r_iiwa_link5_pts, box_pose.matrix(), right_face_nonpenetration_lb, right_face_nonpenetration_ub};

  AddSphereToBody(tree, r_iiwa_link3_idx, Eigen::Vector3d(0, 0, 0), "r_iiwa_link3_pt0", 0.005);
  AddSphereToBody(tree, r_iiwa_link3_idx, Eigen::Vector3d(0.1, 0, 0), "r_iiwa_link3_pt1", 0.005);
  AddSphereToBody(tree, r_iiwa_link3_idx, Eigen::Vector3d(0, 0.1, 0), "r_iiwa_link3_pt2", 0.005);
  AddSphereToBody(tree, r_iiwa_link3_idx, Eigen::Vector3d(0, 0, 0.1), "r_iiwa_link3_pt3", 0.005);
  AddSphereToBody(tree, r_iiwa_link3_idx, Eigen::Vector3d(0, 0.1, 0.23), "r_iiwa_link3_pt4", 0.005);

  Eigen::Vector3d r_iiwa_link3_pts(0, 0.12, 0.23);
  WorldPositionInFrameConstraint r_iiwa_link3_nonpenetration_cnstr{tree, r_iiwa_link3_idx, r_iiwa_link3_pts, box_pose.matrix(), right_face_nonpenetration_lb.col(0), right_face_nonpenetration_ub.col(0)};

  const std::vector<const RigidBodyConstraint*> cnstr_array{
      &box_fixed_pose,
      &left_palm_axis_cnstr,
      &left_palm_contact_constraint,
      &right_palm_axis_cnstr,
      &right_palm_contact_constraint,
      &l_iiwa_link6_nonpenetration_cnstr,
      &l_iiwa_link5_nonpenetration_cnstr,
      &r_iiwa_link5_nonpenetration_cnstr,
      &r_iiwa_link3_nonpenetration_cnstr};

  Eigen::VectorXd q_sol(20);
  int info;
  std::vector<std::string> infeasible_cnstr;
  IKoptions ik_options(tree);

  inverseKin(tree, q, q, cnstr_array.size(), cnstr_array.data(), ik_options,
             &q_sol, &info, &infeasible_cnstr);
  std::cout << info << std::endl;
  return q_sol;
}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
