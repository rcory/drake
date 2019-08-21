#include "drake/examples/planar_gripper/contact_force_qp.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/examples/planar_gripper/gripper_brick_planning_constraint_helper.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace examples {
namespace planar_gripper {
GTEST_TEST(InstantaneousContactForceQPTest, Test) {
  GripperBrickHelper<double> gripper_brick;
  // Set the desired motion to arbitrary value.
  const Eigen::Vector2d p_WB_planned(0.2, 0.1);
  const Eigen::Vector2d v_WB_planned(0.1, -0.5);
  const Eigen::Vector2d a_WB_planned(0.9, 0.2);
  const double theta_planned(0.4);
  const double thetadot_planned(0.9);
  const double thetaddot_planned(1.2);
  const Eigen::Matrix2d Kp1 = Eigen::Vector2d(0.2, 1.5).asDiagonal();
  const Eigen::Matrix2d Kd1 = Eigen::Vector2d(1.2, 0.4).asDiagonal();
  const double Kp2{1.5};
  const double Kd2{0.3};

  // Now solve an IK problem, such that the fingers are in contact with the
  // faces.
  multibody::InverseKinematics ik(gripper_brick.plant());

  const std::map<Finger, BrickFace> finger_face_assignment{
      {Finger::kFinger1, BrickFace::kPosZ},
      {Finger::kFinger2, BrickFace::kNegZ}};

  for (const auto& finger_face : finger_face_assignment) {
    AddFingerTipInContactWithBrickFaceConstraint(
        gripper_brick, finger_face.first, finger_face.second,
        ik.get_mutable_prog(), ik.q(), ik.get_mutable_context(), 0.8, 0);
  }
  // Set q_guess to some arbitrary numbers. zero configuration is a bad initial
  // guess as the fingers are in singularity.
  Eigen::VectorXd q_guess(9);
  q_guess << 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 0.2, 0.3;
  const auto ik_result = solvers::Solve(ik.prog(), q_guess);
  ASSERT_TRUE(ik_result.is_success());
  const Eigen::VectorXd q_ik = ik_result.GetSolution(ik.q());
  Eigen::VectorXd v(9);
  // set v to arbitrary value.
  v << 0.1, 0.2, -0.1, -0.2, 0.2, 0.1, -0.1, 0.2, 0.3;
  auto diagram_context = gripper_brick.diagram().CreateDefaultContext();
  systems::Context<double>* plant_context =
      &(gripper_brick.diagram().GetMutableSubsystemContext(
          gripper_brick.plant(), diagram_context.get()));
  gripper_brick.plant().SetPositions(plant_context, q_ik);
  gripper_brick.plant().SetVelocities(plant_context, v);

  const double weight_a = 2;
  const double weight_thetaddot = 4;
  const double weight_f_Cb = 3;

  InstantaneousContactForceQP qp(
      &gripper_brick, p_WB_planned, v_WB_planned, a_WB_planned, theta_planned,
      thetadot_planned, thetaddot_planned, Kp1, Kd1, Kp2, Kd2, *plant_context,
      weight_a, weight_thetaddot, weight_f_Cb, finger_face_assignment);

  const auto qp_result = solvers::Solve(qp.prog());
  EXPECT_TRUE(qp_result.is_success());

  // Now test if the friction cone constraint is satisfied.
  const std::unordered_map<Finger, Eigen::Vector2d> finger_contact_forces =
      qp.GetFingerContactForceResult(qp_result);
  EXPECT_EQ(finger_contact_forces.size(), finger_face_assignment.size());
  // finger 1 is in contact with +z face.
  Eigen::Vector2d f_Cb1_B = finger_contact_forces.at(Finger::kFinger1);
  const double mu1 =
      gripper_brick.GetFingerTipBrickCoulombFriction(Finger::kFinger1)
          .static_friction();
  EXPECT_LE(f_Cb1_B(1), 0);
  EXPECT_LE(std::abs(f_Cb1_B(0)), mu1 * std::abs(f_Cb1_B(1)));
  // finger 2 is in contact with -z face.
  Eigen::Vector2d f_Cb2_B = finger_contact_forces.at(Finger::kFinger2);
  const double mu2 =
      gripper_brick.GetFingerTipBrickCoulombFriction(Finger::kFinger2)
          .static_friction();
  EXPECT_GE(f_Cb2_B(1), 0);
  EXPECT_LE(std::abs(f_Cb2_B(0)), mu2 * std::abs(f_Cb2_B(1)));

  // Now evaluate the cost by hand.
  // First evaluates the acceleration of the brick.
  const double theta = q_ik(gripper_brick.brick_revolute_x_position_index());
  const double sin_theta = std::sin(theta);
  const double cos_theta = std::cos(theta);
  Eigen::Matrix2d R_WB;
  R_WB << cos_theta, -sin_theta, sin_theta, cos_theta;

  Eigen::Vector2d a_WB =
      Eigen::Vector2d(
          0, -multibody::UniformGravityFieldElement<double>::kDefaultStrength) +
      R_WB * (f_Cb1_B + f_Cb2_B) /
          gripper_brick.brick_frame().body().get_default_mass();
  // Now compute the contact location between the fingers and the brick.
  Eigen::Vector3d p_BF1, p_BF2;
  gripper_brick.plant().SetPositions(plant_context, q_ik);
  gripper_brick.plant().CalcPointsPositions(
      *plant_context, gripper_brick.finger_link2_frame(Finger::kFinger1),
      gripper_brick.p_L2Fingertip(), gripper_brick.brick_frame(), &p_BF1);
  gripper_brick.plant().CalcPointsPositions(
      *plant_context, gripper_brick.finger_link2_frame(Finger::kFinger2),
      gripper_brick.p_L2Fingertip(), gripper_brick.brick_frame(), &p_BF2);
  const Eigen::Vector2d p_BFingertip1 =
      p_BF1.tail<2>() - Eigen::Vector2d(0, gripper_brick.finger_tip_radius());
  const Eigen::Vector2d p_BFingertip2 =
      p_BF2.tail<2>() + Eigen::Vector2d(0, gripper_brick.finger_tip_radius());
  const double thetaddot =
      (p_BFingertip1(0) * f_Cb1_B(1) - p_BFingertip1(1) * f_Cb1_B(0) +
       p_BFingertip2(0) * f_Cb2_B(1) - p_BFingertip2(1) * f_Cb2_B(0)) /
      dynamic_cast<const multibody::RigidBody<double>&>(
          gripper_brick.brick_frame().body())
          .default_rotational_inertia()
          .get_moments()(0);
  // Compute the desired acceleration.
  const Eigen::Vector2d p_WB(
      q_ik(gripper_brick.brick_translate_y_position_index()),
      q_ik(gripper_brick.brick_translate_z_position_index()));
  const Eigen::Vector2d v_WB(
      v(gripper_brick.brick_translate_y_position_index()),
      v(gripper_brick.brick_translate_z_position_index()));
  const Eigen::Vector2d a_WB_des =
      Kp1 * (p_WB_planned - p_WB) + Kd1 * (v_WB_planned - v_WB) + a_WB_planned;
  const double thetadot = v(gripper_brick.brick_revolute_x_position_index());
  const double thetaddot_des = Kp2 * (theta_planned - theta) +
                               Kd2 * (thetadot_planned - thetadot) +
                               thetaddot_planned;
  const double cost_expected =
      weight_a * (a_WB - a_WB_des).squaredNorm() +
      weight_thetaddot * std::pow(thetaddot - thetaddot_des, 2) +
      weight_f_Cb * (f_Cb1_B.squaredNorm() + f_Cb2_B.squaredNorm());
  EXPECT_NEAR(cost_expected, qp_result.get_optimal_cost(), 1E-9);
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
