#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_box_rotation_planner.h"

#include <vector>

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

DualArmsBoxRotationPlanner::DualArmsBoxRotationPlanner(
    RotateBox box_type, const Eigen::Isometry3d& right_kuka_base_pose) {
  tree_ = ConstructDualArmAndBox(box_type, right_kuka_base_pose);
  box_idx_ = tree_->FindBodyIndex("box");
  l_hand_idx_ = tree_->FindBodyIndex("left_iiwa_link_ee_kuka");
  r_hand_idx_ = tree_->FindBodyIndex("right_iiwa_link_ee_kuka");
  for (int i = 0; i < 7; ++i) {
    left_iiwa_link_idx_[i] =
        tree_->FindBodyIndex("left_iiwa_link_" + std::to_string(i));
    right_iiwa_link_idx_[i] =
        tree_->FindBodyIndex("right_iiwa_link_" + std::to_string(i));
  }

  iiwa_link_boundary_pts_[6].resize(3, 4);
  // clang-format off
  iiwa_link_boundary_pts_[6] << 0.04, -0.04, 0, 0,
                                0, 0, 0.04, -0.04,
                                -0.09, -0.09, -0.09, -0.09;
  // clang-format on
  iiwa_link_boundary_pts_[5].resize(3, 4);
  // clang-format off
  iiwa_link_boundary_pts_[5] << 0.07, -0.07, 0, 0,
                                0, 0, 0.07, -0.07,
                                0, 0, 0, 0;
  // clang-format on

  iiwa_link_boundary_pts_[3].resize(3, 1);
  iiwa_link_boundary_pts_[3] << 0, 0.12, 0.23;

  box_normals_ << Eigen::Matrix3d::Identity(), -Eigen::Matrix3d::Identity();
}

Eigen::VectorXd DualArmsBoxRotationPlanner::GrabbingBoxFromTwoSides(
    const Eigen::VectorXd& q, double box_size) const {
  KinematicsCache<double> cache = tree_->CreateKinematicsCache();
  cache.initialize(q);
  tree_->doKinematics(cache);
  const auto box_pose = tree_->relativeTransform(cache, 0, box_idx_);
  const auto right_iiwa_base_pose =
      tree_->relativeTransform(cache, 0, right_iiwa_link_idx_[0]);

  // First determine which face of the box should be grabbed by the left kuka.
  // The normal on this face should be approximately equal to [0;1;0]
  std::array<int, 3> xyz_box_normal_idx = BoxNormalIndicesFacingWorldXYZ(box_pose);
  int box_plusY_face_normal_idx = xyz_box_normal_idx[1];
  const Eigen::Vector3d box_left_face_normal =
      box_normals_.col(box_plusY_face_normal_idx);

  // Now determines which face of the box should be grabbed by the right kuka.
  // The normal on this face should be approximately equal to [0; -1; 0]

  int box_minusY_face_normal_idx = (xyz_box_normal_idx[1] + 3) % 6;
  const Eigen::Vector3d box_right_face_normal =
      box_normals_.col(box_minusY_face_normal_idx);

  int box_front_face_normal_idx;
  if (box_pose.translation()(0) - right_iiwa_base_pose.translation()(0) < 0) {
    box_front_face_normal_idx = xyz_box_normal_idx[0];
  } else {
    box_front_face_normal_idx = (xyz_box_normal_idx[0] + 3) % 6;
  }
  const Eigen::Vector3d box_front_face_normal =
      box_normals_.col(box_front_face_normal_idx);

  // Now determines which face of the box faces upward. The normal on that
  // face should be approximately equal to [0;0;1].
  int box_top_face_normal_idx = xyz_box_normal_idx[2];
  const Eigen::Vector3d box_top_face_normal =
      box_normals_.col(box_top_face_normal_idx);

  // Solve an IK problem on both left and right kuka
  PostureConstraint box_fixed_pose =
      FixRobotJoints(tree_.get(), q, false, false, true);

  // left palm axis aligns with the box, pointing outward.
  WorldGazeDirConstraint left_palm_axis_cnstr(
      tree_.get(), l_hand_idx_, Eigen::Vector3d(0, 0, 1),
      -box_pose.linear() * box_front_face_normal, 0);
  // right palm axis aligns with the box, pointing outward.
  WorldGazeDirConstraint right_palm_axis_cnstr(
      tree_.get(), r_hand_idx_, Eigen::Vector3d(0, 0, 1),
      -box_pose.linear() * box_front_face_normal, 0);

  const double palm_radius = 0.06;
  // left palm contact region
  Eigen::Matrix<double, 3, 4> left_palm_contact_region =
      ((box_size / 2 + palm_radius) * box_left_face_normal) *
      Eigen::RowVector4d::Ones();
  left_palm_contact_region.col(0) += -box_size * 0.2 * box_top_face_normal + box_size * 0.2 * box_front_face_normal;
  left_palm_contact_region.col(1) +=  box_size * 0.2 * box_top_face_normal + box_size * 0.2 * box_front_face_normal;
  left_palm_contact_region.col(2) += -box_size * 0.2 * box_top_face_normal - box_size * 0.2 * box_front_face_normal;
  left_palm_contact_region.col(3) +=  box_size * 0.2 * box_top_face_normal - box_size * 0.2 * box_front_face_normal;

  Eigen::Vector3d left_palm_contact_region_lb =
      left_palm_contact_region.rowwise().minCoeff();
  Eigen::Vector3d left_palm_contact_region_ub =
      left_palm_contact_region.rowwise().maxCoeff();
  WorldPositionInFrameConstraint left_palm_contact_constraint(
      tree_.get(), l_hand_idx_, Eigen::Vector3d::Zero(), box_pose.matrix(),
      left_palm_contact_region_lb, left_palm_contact_region_ub);

  Eigen::Matrix<double, 3, 4> left_face_nonpenetration_lb =
      Eigen::Matrix<double, 3, 4>::Constant(NAN);
  Eigen::Matrix<double, 3, 4> left_face_nonpenetration_ub =
      Eigen::Matrix<double, 3, 4>::Constant(NAN);
  if (box_plusY_face_normal_idx < 3) {
    left_face_nonpenetration_lb.row(box_plusY_face_normal_idx) =
        Eigen::RowVector4d::Constant(box_size / 2);
  } else {
    left_face_nonpenetration_ub.row(box_plusY_face_normal_idx - 3) =
        Eigen::RowVector4d::Constant(-box_size / 2);
  }
  WorldPositionInFrameConstraint l_iiwa_link6_nonpenetration_cnstr(
      tree_.get(), left_iiwa_link_idx_[6], iiwa_link_boundary_pts_[6],
      box_pose.matrix(), left_face_nonpenetration_lb,
      left_face_nonpenetration_ub);

  WorldPositionInFrameConstraint l_iiwa_link5_nonpenetration_cnstr(
      tree_.get(), left_iiwa_link_idx_[5], iiwa_link_boundary_pts_[5],
      box_pose.matrix(), left_face_nonpenetration_lb,
      left_face_nonpenetration_ub);

  // right palm contact region
  Eigen::Matrix<double, 3, 4> right_palm_contact_region =
      ((box_size / 2 + palm_radius) * box_right_face_normal) *
      Eigen::RowVector4d::Ones();
  right_palm_contact_region.col(0) += -box_size * 0.2 * box_top_face_normal + box_size * 0.2 * box_front_face_normal;
  right_palm_contact_region.col(1) +=  box_size * 0.2 * box_top_face_normal + box_size * 0.2 * box_front_face_normal;
  right_palm_contact_region.col(2) += -box_size * 0.2 * box_top_face_normal - box_size * 0.2 * box_front_face_normal;
  right_palm_contact_region.col(3) +=  box_size * 0.2 * box_top_face_normal - box_size * 0.2 * box_front_face_normal;

  // The surface of the right palm should touch the box right face.
  Eigen::Vector3d right_palm_contact_region_lb =
      right_palm_contact_region.rowwise().minCoeff();
  Eigen::Vector3d right_palm_contact_region_ub =
      right_palm_contact_region.rowwise().maxCoeff();
  WorldPositionInFrameConstraint right_palm_contact_constraint(
      tree_.get(), r_hand_idx_, Eigen::Vector3d::Zero(), box_pose.matrix(),
      right_palm_contact_region_lb, right_palm_contact_region_ub);

  // Add the non-penetration constraint on the right face
  Eigen::Matrix<double, 3, 4> right_face_nonpenetration_lb =
      Eigen::Matrix<double, 3, 4>::Constant(NAN);
  Eigen::Matrix<double, 3, 4> right_face_nonpenetration_ub =
      Eigen::Matrix<double, 3, 4>::Constant(NAN);
  if (box_minusY_face_normal_idx < 3) {
    right_face_nonpenetration_lb.row(box_minusY_face_normal_idx) =
        Eigen::RowVector4d::Constant(box_size / 2);
  } else {
    right_face_nonpenetration_ub.row(box_minusY_face_normal_idx - 3) =
        Eigen::RowVector4d::Constant(-box_size / 2);
  }
  WorldPositionInFrameConstraint r_iiwa_link5_nonpenetration_cnstr{
      tree_.get(),
      right_iiwa_link_idx_[5],
      iiwa_link_boundary_pts_[5],
      box_pose.matrix(),
      right_face_nonpenetration_lb,
      right_face_nonpenetration_ub};

  WorldPositionInFrameConstraint r_iiwa_link3_nonpenetration_cnstr{
      tree_.get(),
      right_iiwa_link_idx_[3],
      iiwa_link_boundary_pts_[3],
      box_pose.matrix(),
      right_face_nonpenetration_lb.col(0),
      right_face_nonpenetration_ub.col(0)};

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
  IKoptions ik_options(tree_.get());

  inverseKin(tree_.get(), q, q, cnstr_array.size(), cnstr_array.data(),
             ik_options, &q_sol, &info, &infeasible_cnstr);
  std::cout << info << std::endl;
  return q_sol;
}

Eigen::VectorXd DualArmsBoxRotationPlanner::MoveBox(
    const Eigen::VectorXd& q, const Eigen::Isometry3d& T_WB,
    const std::vector<int>& kuka_box_fixed_link_indices) const {
  // First constrain the box pose.
  const Eigen::Vector3d box_space_xyz_euler = math::rotmat2rpy(T_WB.linear());
  PostureConstraint box_pose_cnstr{tree_.get()};
  Eigen::Matrix<int, 6, 1> box_pos_idx;
  box_pos_idx << 14, 15, 16, 17, 18, 19;
  Eigen::Matrix<double, 6, 1> box_pos_des;
  box_pos_des << T_WB.translation(), box_space_xyz_euler;
  box_pose_cnstr.setJointLimits(box_pos_idx, box_pos_des, box_pos_des);

  // Now constrains some kuka links to move with the box.
  KinematicsCache<double> cache = tree_->CreateKinematicsCache();
  cache.initialize(q);
  tree_->doKinematics(cache);

  std::vector<std::unique_ptr<RigidBodyConstraint>>
      fix_kuka_links_to_box_constraints;
  fix_kuka_links_to_box_constraints.reserve(kuka_box_fixed_link_indices.size());
  // Compute the transform from the iiwa links to the box
  for (int kuka_link_idx : kuka_box_fixed_link_indices) {
    const auto T_boxlink =
        tree_->relativeTransform(cache, box_idx_, kuka_link_idx);
    Eigen::Matrix<double, 7, 1> identity_transform;
    identity_transform << 0, 0, 0, 1, 0, 0, 0;
    const Eigen::Vector3d p_boxlink = tree_->transformPoints(
        cache, Eigen::Vector3d::Zero(), kuka_link_idx, box_idx_);
    const Eigen::Vector4d quat_boxlink = math::rotmat2quat(T_boxlink.linear());
    const Eigen::Vector2d tspan(NAN, NAN);
    std::unique_ptr<RigidBodyConstraint> fix_kuka_link_to_box_position_cnstr =
        std::make_unique<RelativePositionConstraint>(
            tree_.get(), Eigen::Vector3d::Zero(), p_boxlink, p_boxlink,
            kuka_link_idx, box_idx_, identity_transform, tspan);
    std::unique_ptr<RigidBodyConstraint> fix_kuka_link_to_box_orient_cnstr =
        std::make_unique<RelativeQuatConstraint>(
            tree_.get(), kuka_link_idx, box_idx_, quat_boxlink, 0, tspan);
    fix_kuka_links_to_box_constraints.push_back(
        std::move(fix_kuka_link_to_box_position_cnstr));
    fix_kuka_links_to_box_constraints.push_back(
        std::move(fix_kuka_link_to_box_orient_cnstr));
  }

  std::vector<RigidBodyConstraint*> cnstr;
  cnstr.push_back(&box_pose_cnstr);
  for (const auto& fix_kuka_links_to_box_cnstr :
       fix_kuka_links_to_box_constraints) {
    cnstr.push_back(fix_kuka_links_to_box_cnstr.get());
  }
  Eigen::VectorXd q_sol(20);
  int info;
  std::vector<std::string> infeasible_cnstr;
  IKoptions ik_options(tree_.get());

  inverseKin(tree_.get(), q, q, cnstr.size(), cnstr.data(), ik_options, &q_sol,
             &info, &infeasible_cnstr);
  std::cout << info << std::endl;
  return q_sol;
}

Eigen::Matrix3d DualArmsBoxRotationPlanner::BoxNormalFacingWorldXYZ(const Eigen::Isometry3d& box_pose) const {
  std::array<int, 3> box_normal_idx = BoxNormalIndicesFacingWorldXYZ(box_pose);
  Eigen::Matrix3d xyz_box_normals;
  xyz_box_normals << box_normals_.col(box_normal_idx[0]), box_normals_(box_normal_idx[1]), box_normals_(box_normal_idx[2]);
  return xyz_box_normals;
}

std::array<int, 3> DualArmsBoxRotationPlanner::BoxNormalIndicesFacingWorldXYZ(
    const Eigen::Isometry3d& box_pose) const {
  // First determine which face of the box should be grabbed by the left kuka.
  // The normal on this face should be approximately equal to [0;1;0]
  // box_pose.linear() * box_left_face_normal approximately equals to [0;1;0]
  Eigen::Matrix<double, 1, 6> y_face_normal_angle =
      Eigen::Vector3d(0, 1, 0).transpose() * box_pose.linear() * box_normals_;
  int y_face_normal_idx;
  y_face_normal_angle.maxCoeff(&y_face_normal_idx);

  Eigen::Matrix<double, 1, 6> x_face_normal_angle;

  x_face_normal_angle = Eigen::Vector3d(1, 0, 0).transpose() *
      box_pose.linear() * box_normals_;
  int x_face_normal_idx;
  x_face_normal_angle.maxCoeff(&x_face_normal_idx);

  // Now determines which face of the box faces upward. The normal on that
  // face should be approximately equal to [0;0;1].
  Eigen::Matrix<double, 1, 6> z_face_normal_angle =
      Eigen::Vector3d(0, 0, 1).transpose() * box_pose.linear() * box_normals_;
  int z_face_normal_idx;
  z_face_normal_angle.maxCoeff(&z_face_normal_idx);

  std::array<int, 3> box_normal_idx{{x_face_normal_idx, y_face_normal_idx, z_face_normal_idx}};

  return box_normal_idx;
}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
