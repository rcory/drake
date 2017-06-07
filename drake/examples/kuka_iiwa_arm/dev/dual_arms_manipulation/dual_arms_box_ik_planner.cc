#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_box_util.h"

#include "drake/multibody/rigid_body_ik.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/examples/kuka_iiwa_arm/dev/tools/simple_tree_visualizer.h"

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
          Eigen::Vector3d(-kBoxSize * 0.3, kBoxSize / 2 + 0.06, 0),
          Eigen::Vector3d(kBoxSize * 0.3, kBoxSize / 2 + 0.06, kBoxSize * 0.3),
          l_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint l_hand_orient_cnstr(tree, l_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0.1);

      RelativePositionConstraint r_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, -kBoxSize / 2 - 0.07, -kBoxSize * 0.4),
          Eigen::Vector3d(kBoxSize * 0.3, -kBoxSize / 2 - 0.07, -kBoxSize * 0.15),
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
          Eigen::Vector3d(-kBoxSize * 0.3, kBoxSize / 2 + 0.06, 0),
          Eigen::Vector3d(kBoxSize * 0.3, kBoxSize / 2 + 0.06, kBoxSize * 0.3),
          l_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint l_hand_orient_cnstr(tree, l_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0.1);

      RelativePositionConstraint r_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, -kBoxSize / 2 - 0.07, -kBoxSize * 0.4),
          Eigen::Vector3d(kBoxSize * 0.3, -kBoxSize / 2 - 0.07, -kBoxSize * 0.15),
          r_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint r_hand_orient_cnstr(tree, r_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0);

      WorldEulerConstraint box_euler_cnstr(tree, box_idx, Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(M_PI / 5, 0, 0));

      WorldPositionConstraint box_pos_cnstr(tree, box_idx, Eigen::Vector3d(0, -kBoxSize / 2, -kBoxSize/2), Eigen::Vector3d(0.5, 0.5, 0), Eigen::Vector3d(0.5, 0.6, 0));

      const std::vector<const RigidBodyConstraint*> cnstr_array{&l_hand_pos_cnstr, &l_hand_orient_cnstr, &r_hand_pos_cnstr, &r_hand_orient_cnstr, &box_pos_cnstr, &box_euler_cnstr};
      inverseKin(tree, q0, q0, cnstr_array.size(), cnstr_array.data(), ik_options, &q_sol, &info, &infeasible_cnstr);
      break;
    }
    case 2: {
      auto posture_cnstr = FixRobotJoints(tree, q0, true, false, true);

      Eigen::Matrix<double, 7, 1> bTbp = Eigen::Matrix<double, 7, 1>::Zero();
      bTbp(3) = 1.0;
      RelativePositionConstraint r_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, -kBoxSize * 0.5 -0.4, kBoxSize / 2 + 0.07),
          Eigen::Vector3d(kBoxSize * 0.3, -kBoxSize * 0.5 -0.15, kBoxSize / 2 + 0.3),
          r_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint r_hand_orient_cnstr(tree, r_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0.1);
      const std::vector<const RigidBodyConstraint*> cnstr_array{&posture_cnstr, &r_hand_pos_cnstr, &r_hand_orient_cnstr};
      inverseKin(tree, q0, q0, cnstr_array.size(), cnstr_array.data(), ik_options, &q_sol, &info, &infeasible_cnstr);
      break;
    }
    case 3: {
      auto posture_cnstr = FixRobotJoints(tree, q0, true, false, true);

      Eigen::Matrix<double, 7, 1> bTbp = Eigen::Matrix<double, 7, 1>::Zero();
      bTbp(3) = 1.0;
      RelativePositionConstraint r_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, -kBoxSize * 0.3, kBoxSize / 2 + 0.07),
          Eigen::Vector3d(kBoxSize * 0.3, kBoxSize * 0.3, kBoxSize / 2 + 0.07),
          r_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint r_hand_orient_cnstr(tree, r_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0.1);
      const std::vector<const RigidBodyConstraint*> cnstr_array{&posture_cnstr, &r_hand_pos_cnstr, &r_hand_orient_cnstr};
      inverseKin(tree, q0, q0, cnstr_array.size(), cnstr_array.data(), ik_options, &q_sol, &info, &infeasible_cnstr);
      break;
    }
    case 4: {
      Eigen::Matrix<double, 7, 1> bTbp = Eigen::Matrix<double, 7, 1>::Zero();
      bTbp(3) = 1.0;
      RelativePositionConstraint l_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, kBoxSize / 2 + 0.06, 0),
          Eigen::Vector3d(kBoxSize * 0.3, kBoxSize / 2 + 0.06, kBoxSize * 0.3),
          l_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint l_hand_orient_cnstr(tree, l_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0.1);

      RelativePositionConstraint r_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, -kBoxSize * 0.3, kBoxSize / 2 + 0.07),
          Eigen::Vector3d(kBoxSize * 0.3, -kBoxSize * 0.1, kBoxSize / 2 + 0.07),
          r_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint r_hand_orient_cnstr(tree, r_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0);

      WorldEulerConstraint box_euler_cnstr(tree, box_idx, Eigen::Vector3d(M_PI * 0.3, 0, 0), Eigen::Vector3d(M_PI/2, 0, 0));

      WorldPositionConstraint box_pos_cnstr(tree, box_idx, Eigen::Vector3d(0, -kBoxSize / 2, -kBoxSize/2), Eigen::Vector3d(0.5, 0.5, 0), Eigen::Vector3d(0.5, 0.5, 0));
      const std::vector<const RigidBodyConstraint*> cnstr_array{&l_hand_pos_cnstr, &l_hand_orient_cnstr, &r_hand_pos_cnstr, &r_hand_orient_cnstr, &box_pos_cnstr, &box_euler_cnstr};
      inverseKin(tree, q0, q0, cnstr_array.size(), cnstr_array.data(), ik_options, &q_sol, &info, &infeasible_cnstr);
      break;
    }
    case 5: {
      auto posture_cnstr = FixRobotJoints(tree, q0, false, true, true);

      Eigen::Matrix<double, 7, 1> bTbp = Eigen::Matrix<double, 7, 1>::Zero();
      bTbp(3) = 1.0;
      RelativePositionConstraint l_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, kBoxSize * 0.5 + 0.1, -kBoxSize / 2 - 0.3),
          Eigen::Vector3d(kBoxSize * 0.3, kBoxSize * 0.5 + 0.3, -kBoxSize / 2 - 0.1),
          l_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint l_hand_orient_cnstr(tree, l_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0);
      const std::vector<const RigidBodyConstraint*> cnstr_array{&posture_cnstr, &l_hand_pos_cnstr, &l_hand_orient_cnstr};
      inverseKin(tree, q0, q0, cnstr_array.size(), cnstr_array.data(), ik_options, &q_sol, &info, &infeasible_cnstr);
      break;
    }
    case 6: {
      auto posture_cnstr = FixRobotJoints(tree, q0, false, true, true);

      Eigen::Matrix<double, 7, 1> bTbp = Eigen::Matrix<double, 7, 1>::Zero();
      bTbp(3) = 1.0;
      RelativePositionConstraint l_hand_pos_cnstr(
          tree, Eigen::Vector3d(0, 0, -0.2),
          Eigen::Vector3d(-kBoxSize * 0.3, kBoxSize * 0.1, -kBoxSize / 2 - 0.07),
          Eigen::Vector3d(kBoxSize * 0.3, kBoxSize * 0.3, -kBoxSize / 2 - 0.07),
          l_hand_idx, box_idx, bTbp, DrakeRigidBodyConstraint::default_tspan);
      RelativeGazeDirConstraint l_hand_orient_cnstr(tree, l_hand_idx, box_idx, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 0), 0);
      const std::vector<const RigidBodyConstraint*> cnstr_array{&posture_cnstr, &l_hand_pos_cnstr, &l_hand_orient_cnstr};
      inverseKin(tree, q0, q0, cnstr_array.size(), cnstr_array.data(), ik_options, &q_sol, &info, &infeasible_cnstr);
      break;
    }
  }
  std::cout << "IK info: " << info << std::endl;
  return q_sol;
};

int DoMain() {
  auto tree = ConstructDualArmAndBox();
  std::vector<Eigen::VectorXd> q;
  q.push_back(PlanDualArmsBoxPosture(tree.get(), 0, Eigen::VectorXd::Zero(20)));
  for (int i = 1; i < 7; ++i) {
    q.push_back(PlanDualArmsBoxPosture(tree.get(), i, q[i - 1]));
  }
  drake::lcm::DrakeLcm lcm;
  tools::SimpleTreeVisualizer visualizer(*tree, &lcm);
  for (int i = 0; i < 7; ++i) {
    visualizer.visualize(q[i]);
  }
  return 0;
}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main() {
  return drake::examples::kuka_iiwa_arm::DoMain();
}