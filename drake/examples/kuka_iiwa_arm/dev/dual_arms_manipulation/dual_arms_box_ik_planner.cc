#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_box_util.h"

#include "drake/multibody/rigid_body_ik.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/examples/kuka_iiwa_arm/dev/tools/simple_tree_visualizer.h"

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
PostureConstraint FixRobotJoints(RigidBodyTreed* tree, const Eigen::VectorXd& q, bool fix_kuka1, bool fix_kuka2, bool fix_box) {
  PostureConstraint posture_cnstr(tree);
  Eigen::VectorXd q_lb = Eigen::Matrix<double, 20, 1>::Constant(-std::numeric_limits<double>::infinity());
  Eigen::VectorXd q_ub = Eigen::Matrix<double, 20, 1>::Constant(std::numeric_limits<double>::infinity());
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
/**
 * Plan the posture of the dual arms and box
 * @param posture_id 0, 1, 2, 3, .... Each ID represent a desired keyframe posture.
 * @return The posture for kuka1, kuka2, and the box
 */
Eigen::VectorXd PlanDualArmsBoxPosture(RigidBodyTreed* tree, int posture_id, const Eigen::VectorXd& q0) {
  const double kBoxSize = 0.56;
  int l_hand_idx = tree->FindBodyIndex("left_iiwa_link_ee_kuka");
  int r_hand_idx = tree->FindBodyIndex("right_iiwa_link_ee_kuka");
  int box_idx = tree->FindBodyIndex("box");

  IKoptions ik_options(tree);
  Eigen::VectorXd q_sol(20);
  int info;
  std::vector<std::string> infeasible_cnstr;
  switch (posture_id) {
    case 0: {
      Eigen::Matrix<double, 7, 1> bTbp = Eigen::Matrix<double, 7, 1>::Zero();
      bTbp(3) = 1.0;
      RelativePositionConstraint l_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, kBoxSize / 2 + 0.12, 0),
          Eigen::Vector3d(kBoxSize * 0.3, kBoxSize / 2 + 0.12, kBoxSize * 0.3),
          l_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint l_hand_orient_cnstr(tree, l_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0.1);

      RelativePositionConstraint r_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, -kBoxSize / 2 - 0.12, -kBoxSize * 0.4),
          Eigen::Vector3d(kBoxSize * 0.3, -kBoxSize / 2 - 0.12, -kBoxSize * 0.15),
          r_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint r_hand_orient_cnstr(tree, r_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0);

      WorldEulerConstraint box_euler_cnstr(tree, box_idx, Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0));

      WorldPositionConstraint box_pos_cnstr(tree, box_idx, Eigen::Vector3d::Zero(), Eigen::Vector3d(0.5, 0.5, kBoxSize / 2), Eigen::Vector3d(0.5, 0.5, kBoxSize / 2));

      const std::vector<const RigidBodyConstraint*> cnstr_array{&l_hand_pos_cnstr, &l_hand_orient_cnstr, &r_hand_pos_cnstr, &r_hand_orient_cnstr, &box_pos_cnstr, &box_euler_cnstr};
      inverseKin(tree, q0, q0, cnstr_array.size(), cnstr_array.data(), ik_options, &q_sol, &info, &infeasible_cnstr);
      break;
    }
    case 1: {
      Eigen::Matrix<double, 7, 1> bTbp = Eigen::Matrix<double, 7, 1>::Zero();
      bTbp(3) = 1.0;
      RelativePositionConstraint l_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, kBoxSize / 2 + 0.09, 0),
          Eigen::Vector3d(kBoxSize * 0.3, kBoxSize / 2 + 0.09, kBoxSize * 0.3),
          l_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint l_hand_orient_cnstr(tree, l_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0.1);

      RelativePositionConstraint r_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, -kBoxSize / 2 - 0.08, -kBoxSize * 0.4),
          Eigen::Vector3d(kBoxSize * 0.3, -kBoxSize / 2 - 0.08, -kBoxSize * 0.15),
          r_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint r_hand_orient_cnstr(tree, r_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0);

      WorldEulerConstraint box_euler_cnstr(tree, box_idx, Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0));

      WorldPositionConstraint box_pos_cnstr(tree, box_idx, Eigen::Vector3d::Zero(), Eigen::Vector3d(0.5, 0.5, kBoxSize / 2), Eigen::Vector3d(0.5, 0.5, kBoxSize / 2));

      const std::vector<const RigidBodyConstraint*> cnstr_array{&l_hand_pos_cnstr, &l_hand_orient_cnstr, &r_hand_pos_cnstr, &r_hand_orient_cnstr, &box_pos_cnstr, &box_euler_cnstr};
      inverseKin(tree, q0, q0, cnstr_array.size(), cnstr_array.data(), ik_options, &q_sol, &info, &infeasible_cnstr);
      break;
    }
    case 2: {
      Eigen::Matrix<double, 7, 1> bTbp = Eigen::Matrix<double, 7, 1>::Zero();
      bTbp(3) = 1.0;
      RelativePositionConstraint l_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, kBoxSize / 2 + 0.09, 0),
          Eigen::Vector3d(kBoxSize * 0.3, kBoxSize / 2 + 0.09, kBoxSize * 0.3),
          l_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint l_hand_orient_cnstr(tree, l_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0.1);

      RelativePositionConstraint r_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, -kBoxSize / 2 - 0.08, -kBoxSize * 0.4),
          Eigen::Vector3d(kBoxSize * 0.3, -kBoxSize / 2 - 0.08, -kBoxSize * 0.15),
          r_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint r_hand_orient_cnstr(tree, r_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0);

      WorldEulerConstraint box_euler_cnstr(tree, box_idx, Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(M_PI / 5, 0, 0));

      WorldPositionConstraint box_pos_cnstr(tree, box_idx, Eigen::Vector3d(0, -kBoxSize / 2, -kBoxSize/2), Eigen::Vector3d(0.5, 0.5, 0), Eigen::Vector3d(0.5, 0.5, 0));

      const std::vector<const RigidBodyConstraint*> cnstr_array{&l_hand_pos_cnstr, &l_hand_orient_cnstr, &r_hand_pos_cnstr, &r_hand_orient_cnstr, &box_pos_cnstr, &box_euler_cnstr};
      inverseKin(tree, q0, q0, cnstr_array.size(), cnstr_array.data(), ik_options, &q_sol, &info, &infeasible_cnstr);
      break;
    }
    case 3: {
      auto posture_cnstr = FixRobotJoints(tree, q0, true, false, true);

      Eigen::Matrix<double, 7, 1> bTbp = Eigen::Matrix<double, 7, 1>::Zero();
      bTbp(3) = 1.0;
      RelativePositionConstraint r_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, -kBoxSize * 0.5 -0.4, kBoxSize  * 0.4),
          Eigen::Vector3d(kBoxSize * 0.3, -kBoxSize * 0.5 -0.2, kBoxSize  * 0.5),
          r_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint r_hand_orient_cnstr(tree, r_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0.1);
      const std::vector<const RigidBodyConstraint*> cnstr_array{&posture_cnstr, &r_hand_pos_cnstr, &r_hand_orient_cnstr};
      inverseKin(tree, q0, q0, cnstr_array.size(), cnstr_array.data(), ik_options, &q_sol, &info, &infeasible_cnstr);
      break;
    }
    case 4: {
      auto posture_cnstr = FixRobotJoints(tree, q0, true, false, true);

      Eigen::Matrix<double, 7, 1> bTbp = Eigen::Matrix<double, 7, 1>::Zero();
      bTbp(3) = 1.0;
      RelativePositionConstraint r_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, -kBoxSize * 0.5, kBoxSize  * 0.5 + 0.25),
          Eigen::Vector3d(kBoxSize * 0.3, -kBoxSize * 0.4, kBoxSize  * 0.5 + 0.4),
          r_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint r_hand_orient_cnstr(tree, r_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0.1);
      const std::vector<const RigidBodyConstraint*> cnstr_array{&posture_cnstr, &r_hand_pos_cnstr, &r_hand_orient_cnstr};
      inverseKin(tree, q0, q0, cnstr_array.size(), cnstr_array.data(), ik_options, &q_sol, &info, &infeasible_cnstr);
      break;
    }
    case 5: {
      auto posture_cnstr = FixRobotJoints(tree, q0, true, false, true);

      Eigen::Matrix<double, 7, 1> bTbp = Eigen::Matrix<double, 7, 1>::Zero();
      bTbp(3) = 1.0;
      RelativePositionConstraint r_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, -kBoxSize * 0.3, kBoxSize / 2 + 0.12),
          Eigen::Vector3d(kBoxSize * 0.3, kBoxSize * 0.3, kBoxSize / 2 + 0.12),
          r_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint r_hand_orient_cnstr(tree, r_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0.1);
      const std::vector<const RigidBodyConstraint*> cnstr_array{&posture_cnstr, &r_hand_pos_cnstr, &r_hand_orient_cnstr};
      inverseKin(tree, q0, q0, cnstr_array.size(), cnstr_array.data(), ik_options, &q_sol, &info, &infeasible_cnstr);
      break;
    }
    case 6: {
      auto cache = tree->doKinematics(q0);
      //auto l_hand_pose0 = tree->CalcBodyPoseInWorldFrame(cache, tree->get_body(l_hand_idx));
      auto r_hand_pose0 = tree->CalcBodyPoseInWorldFrame(cache, tree->get_body(r_hand_idx));
      auto box_pose0 = tree->CalcBodyPoseInWorldFrame(cache, tree->get_body(box_idx));
      //Eigen::Isometry3d l_hand_in_box_pose0 = box_pose0.inverse() * l_hand_pose0;
      Eigen::Isometry3d r_hand_in_box_pose0 = box_pose0.inverse() * r_hand_pose0;
      Eigen::Matrix<double, 7, 1> bTbp = Eigen::Matrix<double, 7, 1>::Zero();
      bTbp(3) = 1.0;
      RelativePositionConstraint l_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, kBoxSize / 2 + 0.08, 0),
          Eigen::Vector3d(kBoxSize * 0.3, kBoxSize / 2 + 0.08, kBoxSize * 0.3),
          l_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint l_hand_orient_cnstr(tree, l_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0.1);

      RelativePositionConstraint r_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, 0),
          r_hand_in_box_pose0.translation() + Eigen::Vector3d(0, -0.05, 0), r_hand_in_box_pose0.translation() + Eigen::Vector3d(0, 0.05, 0.05),
          r_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint r_hand_orient_cnstr(tree, r_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0);

      WorldEulerConstraint box_euler_cnstr(tree, box_idx, Eigen::Vector3d(M_PI * 0.3, 0, 0), Eigen::Vector3d(M_PI/2, 0, 0));

      WorldPositionConstraint box_pos_cnstr(tree, box_idx, Eigen::Vector3d(0, -kBoxSize / 2, -kBoxSize/2), Eigen::Vector3d(0.5, 0.5, 0), Eigen::Vector3d(0.5, 0.5, 0));
      const std::vector<const RigidBodyConstraint*> cnstr_array{&l_hand_pos_cnstr, &l_hand_orient_cnstr, &r_hand_pos_cnstr, &r_hand_orient_cnstr, &box_pos_cnstr, &box_euler_cnstr};
      inverseKin(tree, q0, q0, cnstr_array.size(), cnstr_array.data(), ik_options, &q_sol, &info, &infeasible_cnstr);
      break;
    }
    case 7: {
      auto posture_cnstr = FixRobotJoints(tree, q0, false, true, true);

      Eigen::Matrix<double, 7, 1> bTbp = Eigen::Matrix<double, 7, 1>::Zero();
      bTbp(3) = 1.0;
      RelativePositionConstraint l_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, kBoxSize * 0.5 + 0.2, -kBoxSize * 0.5),
          Eigen::Vector3d(kBoxSize * 0.3, kBoxSize * 0.5 + 0.4, -kBoxSize * 0.4),
          l_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint l_hand_orient_cnstr(tree, l_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0);
      const std::vector<const RigidBodyConstraint*> cnstr_array{&posture_cnstr, &l_hand_pos_cnstr, &l_hand_orient_cnstr};
      inverseKin(tree, q0, q0, cnstr_array.size(), cnstr_array.data(), ik_options, &q_sol, &info, &infeasible_cnstr);
      break;
    }
    case 8: {
      auto posture_cnstr = FixRobotJoints(tree, q0, false, true, true);

      Eigen::Matrix<double, 7, 1> bTbp = Eigen::Matrix<double, 7, 1>::Zero();
      bTbp(3) = 1.0;
      RelativePositionConstraint l_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, kBoxSize * 0.4 , -kBoxSize * 0.5 - 0.4),
          Eigen::Vector3d(kBoxSize * 0.3, kBoxSize * 0.5, -kBoxSize * 0.5 - 0.3),
          l_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint l_hand_orient_cnstr(tree, l_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0);
      const std::vector<const RigidBodyConstraint*> cnstr_array{&posture_cnstr, &l_hand_pos_cnstr, &l_hand_orient_cnstr};
      inverseKin(tree, q0, q0, cnstr_array.size(), cnstr_array.data(), ik_options, &q_sol, &info, &infeasible_cnstr);
      break;
    }
    case 9: {
      auto posture_cnstr = FixRobotJoints(tree, q0, false, true, true);

      Eigen::Matrix<double, 7, 1> bTbp = Eigen::Matrix<double, 7, 1>::Zero();
      bTbp(3) = 1.0;
      RelativePositionConstraint l_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, kBoxSize * 0.2, -kBoxSize / 2 - 0.07),
          Eigen::Vector3d(kBoxSize * 0.3, kBoxSize * 0.4, -kBoxSize / 2 - 0.07),
          l_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint l_hand_orient_cnstr(tree, l_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0);
      const std::vector<const RigidBodyConstraint*> cnstr_array{&posture_cnstr, &l_hand_pos_cnstr, &l_hand_orient_cnstr};
      inverseKin(tree, q0, q0, cnstr_array.size(), cnstr_array.data(), ik_options, &q_sol, &info, &infeasible_cnstr);
      break;
    }
    case 10: {
      auto cache = tree->doKinematics(q0);
      auto l_hand_pose0 = tree->CalcBodyPoseInWorldFrame(cache, tree->get_body(l_hand_idx));
      auto r_hand_pose0 = tree->CalcBodyPoseInWorldFrame(cache, tree->get_body(r_hand_idx));
      auto box_pose0 = tree->CalcBodyPoseInWorldFrame(cache, tree->get_body(box_idx));
      Eigen::Isometry3d l_hand_in_box_pose0 = box_pose0.inverse() * l_hand_pose0;
      Eigen::Isometry3d r_hand_in_box_pose0 = box_pose0.inverse() * r_hand_pose0;
      Eigen::Matrix<double, 7, 1> bTbp = Eigen::Matrix<double, 7, 1>::Zero();
      bTbp(3) = 1.0;
      RelativePositionConstraint l_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, 0),
          l_hand_in_box_pose0.translation() + Eigen::Vector3d(-0.15, 0, 0.02), l_hand_in_box_pose0.translation() + Eigen::Vector3d(-0.1, 0, 0.03),
          l_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint l_hand_orient_cnstr(tree, l_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0.1);

      RelativePositionConstraint r_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, 0),
          r_hand_in_box_pose0.translation() + Eigen::Vector3d(0, -0.03, -0.04), r_hand_in_box_pose0.translation() + Eigen::Vector3d(0, 0.03, -0.03),
          r_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint r_hand_orient_cnstr(tree, r_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0);

      WorldEulerConstraint box_euler_cnstr(tree, box_idx, Eigen::Vector3d(M_PI * 0.5, 0, 0), Eigen::Vector3d(M_PI * 0.5, 0, 0));

      WorldPositionConstraint box_pos_cnstr(tree, box_idx, Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0.5, 0.45, kBoxSize / 2), Eigen::Vector3d(0.5, 0.45, kBoxSize / 2));
      const std::vector<const RigidBodyConstraint*> cnstr_array{&l_hand_pos_cnstr, &l_hand_orient_cnstr, &r_hand_pos_cnstr, &r_hand_orient_cnstr, &box_pos_cnstr, &box_euler_cnstr};
      inverseKin(tree, q0, q0, cnstr_array.size(), cnstr_array.data(), ik_options, &q_sol, &info, &infeasible_cnstr);
      break;
    }
    case 11: {
      auto cache = tree->doKinematics(q0);
      auto l_hand_pose0 = tree->CalcBodyPoseInWorldFrame(cache, tree->get_body(l_hand_idx));
      auto r_hand_pose0 = tree->CalcBodyPoseInWorldFrame(cache, tree->get_body(r_hand_idx));
      auto box_pose0 = tree->CalcBodyPoseInWorldFrame(cache, tree->get_body(box_idx));
      Eigen::Isometry3d l_hand_in_box_pose0 = box_pose0.inverse() * l_hand_pose0;
      Eigen::Isometry3d r_hand_in_box_pose0 = box_pose0.inverse() * r_hand_pose0;
      Eigen::Matrix<double, 7, 1> bTbp = Eigen::Matrix<double, 7, 1>::Zero();
      bTbp(3) = 1.0;
      RelativePositionConstraint l_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, 0),
          l_hand_in_box_pose0.translation() + Eigen::Vector3d(0, 0, -0.08), l_hand_in_box_pose0.translation() + Eigen::Vector3d(0, 0, -0.073),
          l_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint l_hand_orient_cnstr(tree, l_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0.1);

      RelativePositionConstraint r_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, 0),
          r_hand_in_box_pose0.translation() + Eigen::Vector3d(0, -0.02, 0.02), r_hand_in_box_pose0.translation() + Eigen::Vector3d(0, 0.02, 0.02),
          r_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint r_hand_orient_cnstr(tree, r_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0);

      WorldEulerConstraint box_euler_cnstr(tree, box_idx, Eigen::Vector3d(M_PI * 0.5, 0, 0), Eigen::Vector3d(M_PI * 0.5, 0, 0));

      WorldPositionConstraint box_pos_cnstr(tree, box_idx, Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0.5, 0.53, kBoxSize / 2), Eigen::Vector3d(0.5, 0.53, kBoxSize / 2));
      const std::vector<const RigidBodyConstraint*> cnstr_array{&l_hand_pos_cnstr, &l_hand_orient_cnstr, &r_hand_pos_cnstr, &r_hand_orient_cnstr, &box_pos_cnstr, &box_euler_cnstr};
      inverseKin(tree, q0, q0, cnstr_array.size(), cnstr_array.data(), ik_options, &q_sol, &info, &infeasible_cnstr);
      break;
    }
  }
  std::cout << "IK info: " << info << std::endl;
  return q_sol;
};

void RemoveFileIfExist(const std::string& file_name) {
  std::ifstream file(file_name);
  if (file) {
    if (remove(file_name.c_str()) != 0) {
      throw std::runtime_error("Error deleting file " + file_name);
    }
  }
  file.close();
}

int DoMain() {
  auto tree = ConstructDualArmAndBox();
  std::vector<Eigen::VectorXd> q;
  q.push_back(Eigen::VectorXd::Zero(20));
  for (int i = 1; i < 13; ++i) {
    q.push_back(PlanDualArmsBoxPosture(tree.get(), i - 1, q[i - 1]));
  }
  drake::lcm::DrakeLcm lcm;
  tools::SimpleTreeVisualizer visualizer(*tree, &lcm);
  for (int i = 0; i < static_cast<int>(q.size()); ++i) {
    visualizer.visualize(q[i]);
  }

  std::string output_file_name = "keyframes.txt";
  RemoveFileIfExist(output_file_name);
  std::fstream output_file;
  output_file.open(output_file_name, std::ios::app | std::ios::out);
  if (output_file.is_open()) {
    for (int i = 0; i < static_cast<int>(q.size()); ++i) {
      output_file << "{\n";
      output_file << "  \"allow_mirror\": true,\n";
      output_file << "  \"description\": \"\",\n";
      output_file << "  \"joints\": {\n";
      for (int j = 0; j < 7; ++j) {
        output_file << "    \"left_iiwa_joint_" + std::to_string(j + 1) + "\": " << q[i](j) << ",\n";
      }
      for (int j = 0; j < 7; ++j) {
        std::string trailing_charater = j == 6 ? "":",";
        output_file << "    \"right_iiwa_joint_" + std::to_string(j + 1) + "\": " << q[i](j + 7) << trailing_charater <<"\n";
      }
      output_file << "  },\n";
      output_file << "  \"name\": \"pose_L" + std::to_string(i) << "\",\n";
      output_file << "  \"nominal_handedness\": \"left\"\n";
      output_file << "},\n";
      }
    output_file.close();
  }
  return 0;
}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main() {
  return drake::examples::kuka_iiwa_arm::DoMain();
}