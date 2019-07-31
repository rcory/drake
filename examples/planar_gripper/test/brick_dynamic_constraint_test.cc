#include "drake/examples/planar_gripper/brick_dynamic_constraint.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/math/compute_numerical_gradient.h"

namespace drake {
namespace examples {
namespace planar_gripper {
class BrickDynamicConstraintTest : public ::testing::Test {
 public:
  BrickDynamicConstraintTest()
      : gripper_brick_(),
        finger_faces_{{{Finger::kFinger1, BrickFace::kNegY},
                       {Finger::kFinger2, BrickFace::kPosZ}}},
        brick_lid_friction_force_magnitude_{1.5},
        brick_lid_friction_torque_magnitude_{2} {}

 protected:
  GripperBrickHelper<double> gripper_brick_;
  std::map<Finger, BrickFace> finger_faces_;
  double brick_lid_friction_force_magnitude_;
  double brick_lid_friction_torque_magnitude_;
};

TEST_F(BrickDynamicConstraintTest, TestBrickTotalWrenchEvaluator) {
  auto diagram_context = gripper_brick_.diagram().CreateDefaultContext();
  systems::Context<double>* plant_context =
      &(gripper_brick_.diagram().GetMutableSubsystemContext(
          gripper_brick_.plant(), diagram_context.get()));
  BrickTotalWrenchEvaluator evaluator(&gripper_brick_, plant_context,
                                      finger_faces_,
                                      brick_lid_friction_force_magnitude_,
                                      brick_lid_friction_torque_magnitude_);
  EXPECT_EQ(evaluator.num_outputs(), 3);
  EXPECT_EQ(evaluator.num_vars(),
            gripper_brick_.plant().num_positions() + 3 + 2 * 2);

  // Test the eval function with arbitrary input value.
  Eigen::VectorXd q(9);
  q << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
  double v_brick_translation_y = 0.45;
  double v_brick_translation_z = 0.55;
  double v_brick_rotation_x = 0.32;
  Matrix2<double> f_FB_B;
  f_FB_B << 1.3, 0.4, -2.5, 0.7;
  Eigen::VectorXd x_val;
  evaluator.ComposeX<double>(q, v_brick_translation_y, v_brick_translation_z,
                             v_brick_rotation_x, f_FB_B, &x_val);
  // Set the gradient to arbitrary value.
  Eigen::MatrixXd x_grad(x_val.rows(), 2);
  for (int i = 0; i < x_val.rows(); ++i) {
    x_grad(i, 0) = i * 2 + 1;
    x_grad(i, 1) = std::sin(x_val(i));
  }
  const auto x_autodiff =
      math::initializeAutoDiffGivenGradientMatrix(x_val, x_grad);
  AutoDiffVecXd y_autodiff;
  evaluator.Eval(x_autodiff, &y_autodiff);

  const Eigen::Vector3d total_wrench_val =
      math::autoDiffToValueMatrix(y_autodiff);

  Eigen::Vector2d total_force_expected =
      gripper_brick_.brick_frame().body().get_default_mass() *
      Eigen::Vector2d(0, -9.81);
  total_force_expected +=
      -Eigen::Vector2d(v_brick_translation_y, v_brick_translation_z)
           .normalized() *
      brick_lid_friction_force_magnitude_;
  const double theta = q(gripper_brick_.brick_revolute_x_position_index());
  Eigen::Matrix2d R_WB;
  R_WB << std::cos(theta), -std::sin(theta), std::sin(theta), std::cos(theta);
  total_force_expected += R_WB * f_FB_B.rowwise().sum();
  // This tolerance is large, because we implement a smoothed version
  // of sqrt function, when computing the unit length vector
  // along the brick velocity.
  double tol = 1E-5;
  EXPECT_TRUE(
      CompareMatrices(total_wrench_val.head<2>(), total_force_expected, tol));
  double total_torque = v_brick_rotation_x > 0
                            ? -brick_lid_friction_torque_magnitude_
                            : brick_lid_friction_torque_magnitude_;
  Eigen::Vector3d p_BF1, p_BF2;
  gripper_brick_.plant().SetPositions(plant_context, q);
  gripper_brick_.plant().CalcPointsPositions(
      *plant_context, gripper_brick_.finger_link2_frame(Finger::kFinger1),
      gripper_brick_.p_L2Tip(), gripper_brick_.brick_frame(), &p_BF1);
  gripper_brick_.plant().CalcPointsPositions(
      *plant_context, gripper_brick_.finger_link2_frame(Finger::kFinger2),
      gripper_brick_.p_L2Tip(), gripper_brick_.brick_frame(), &p_BF2);
  // The witness point on finger 1 is Cf1, and the witness point on finger 2 is
  // Cf2.
  const Eigen::Vector3d p_BCf1 =
      p_BF1 + Eigen::Vector3d(0, gripper_brick_.finger_tip_radius(), 0);
  const Eigen::Vector3d p_BCf2 =
      p_BF2 + Eigen::Vector3d(0, 0, -gripper_brick_.finger_tip_radius());
  const Eigen::Vector3d total_contact_torque =
      p_BCf1.cross(Eigen::Vector3d(0, f_FB_B(0, 0), f_FB_B(1, 0))) +
      p_BCf2.cross(Eigen::Vector3d(0, f_FB_B(0, 1), f_FB_B(1, 1)));
  total_torque += total_contact_torque(0);
  EXPECT_NEAR(total_wrench_val(2), total_torque, tol);

  // Now check the gradient.
  std::function<void(const Eigen::Ref<const Eigen::VectorXd>&,
                     Eigen::VectorXd*)>
      evaluator_double =
          [&evaluator](const Eigen::Ref<const Eigen::VectorXd>& x,
                       Eigen::VectorXd* y) { return evaluator.Eval(x, y); };
  const auto J = math::ComputeNumericalGradient(evaluator_double, x_val);
  EXPECT_TRUE(CompareMatrices(math::autoDiffToGradientMatrix(y_autodiff),
                              J * x_grad, 1E-5));
}

TEST_F(BrickDynamicConstraintTest, TestBrickDynamicBackwardEulerConstraint) {
  auto diagram_context = gripper_brick_.diagram().CreateDefaultContext();
  systems::Context<double>* plant_context =
      &(gripper_brick_.diagram().GetMutableSubsystemContext(
          gripper_brick_.plant(), diagram_context.get()));
  const BrickDynamicBackwardEulerConstraint constraint(
      &gripper_brick_, plant_context, finger_faces_,
      brick_lid_friction_force_magnitude_,
      brick_lid_friction_torque_magnitude_);

  EXPECT_EQ(constraint.num_constraints(), 3);
  EXPECT_EQ(constraint.num_vars(),
            gripper_brick_.plant().num_positions() + 6 + 4 + 1);
  EXPECT_TRUE(
      CompareMatrices(constraint.lower_bound(), Eigen::Vector3d::Zero()));
  EXPECT_TRUE(
      CompareMatrices(constraint.upper_bound(), Eigen::Vector3d::Zero()));

  BrickTotalWrenchEvaluator evaluator(&gripper_brick_, plant_context,
                                      finger_faces_,
                                      brick_lid_friction_force_magnitude_,
                                      brick_lid_friction_torque_magnitude_);

  // Evaluate the constraint with arbitrary input.
  Eigen::VectorXd q(9);
  q << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
  const double v_brick_r_rotation_x = 0.4;
  const double v_brick_r_translation_y = 1.2;
  const double v_brick_r_translation_z = 2.5;
  const double v_brick_l_rotation_x = -0.5;
  const double v_brick_l_translation_y = 2.4;
  const double v_brick_l_translation_z = 1.3;
  Eigen::Matrix2d f_FB_B;
  f_FB_B << 1.3, -2.4, 0.5, 1.2;
  double dt = 1.1;
  Eigen::VectorXd constraint_x;
  constraint.ComposeX<double>(q, v_brick_r_translation_y,
                              v_brick_r_translation_z, v_brick_r_rotation_x,
                              v_brick_l_translation_y, v_brick_l_translation_z,
                              v_brick_l_rotation_x, f_FB_B, dt, &constraint_x);
  Eigen::VectorXd constraint_y;
  constraint.Eval(constraint_x, &constraint_y);
  Eigen::VectorXd evaluator_x;
  evaluator.ComposeX<double>(q, v_brick_r_translation_y,
                             v_brick_r_translation_z, v_brick_r_rotation_x,
                             f_FB_B, &evaluator_x);
  Eigen::VectorXd total_wrench;
  evaluator.Eval(evaluator_x, &total_wrench);
  Eigen::Vector3d y_expected;
  y_expected.head<2>() =
      gripper_brick_.brick_frame().body().get_default_mass() *
          Eigen::Vector2d(v_brick_r_translation_y - v_brick_l_translation_y,
                          v_brick_r_translation_z - v_brick_l_translation_z) -
      total_wrench.head<2>() * dt;
  const double I = gripper_brick_.brick_frame()
                       .body()
                       .CalcSpatialInertiaInBodyFrame(*plant_context)
                       .CalcRotationalInertia()
                       .get_moments()(0);
  y_expected(2) =
      I * (v_brick_r_rotation_x - v_brick_l_rotation_x) - total_wrench(2) * dt;
  EXPECT_TRUE(CompareMatrices(constraint_y, y_expected, 1E-13));
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
