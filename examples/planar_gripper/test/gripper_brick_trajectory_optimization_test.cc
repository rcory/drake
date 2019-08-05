#include "drake/examples/planar_gripper/gripper_brick_trajectory_optimization.h"

#include <limits>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/examples/planar_gripper/brick_dynamic_constraint.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace examples {
namespace planar_gripper {
const double kInf = std::numeric_limits<double>::infinity();

GTEST_TEST(GripperBrickTrajectoryOptimizationTest, TestConstructor) {
  GripperBrickHelper<double> gripper_brick;
  int nT = 9;
  std::map<Finger, BrickFace> initial_contact(
      {{Finger::kFinger1, BrickFace::kNegY},
       {Finger::kFinger2, BrickFace::kNegZ},
       {Finger::kFinger3, BrickFace::kPosY}});
  std::vector<FingerTransition> finger_transitions;
  finger_transitions.emplace_back(5, 7, Finger::kFinger2, BrickFace::kPosY);
  finger_transitions.emplace_back(1, 3, Finger::kFinger1, BrickFace::kNegZ);

  const double brick_lid_friction_force_magnitude = 1.5;
  const double brick_lid_friction_torque_magnitude = 2;

  const double depth = 1E-3;
  const double friction_cone_shrink_factor = 1;

  GripperBrickTrajectoryOptimization dut(
      &gripper_brick, nT, initial_contact, finger_transitions,
      brick_lid_friction_force_magnitude, brick_lid_friction_torque_magnitude,
      GripperBrickTrajectoryOptimization::Options(
          0.8,
          GripperBrickTrajectoryOptimization::IntegrationMethod::kBackwardEuler,
          0.05 * M_PI, 0.02, depth, friction_cone_shrink_factor));

  EXPECT_EQ(dut.finger_face_contacts()[0], initial_contact);
  EXPECT_EQ(dut.finger_face_contacts()[1], initial_contact);
  std::map<Finger, BrickFace> finger_face_contacts_expected = {
      {Finger::kFinger2, BrickFace::kNegZ},
      {Finger::kFinger3, BrickFace::kPosY}};
  EXPECT_EQ(dut.finger_face_contacts()[2], finger_face_contacts_expected);
  finger_face_contacts_expected.emplace(Finger::kFinger1, BrickFace::kNegZ);
  EXPECT_EQ(dut.finger_face_contacts()[3], finger_face_contacts_expected);
  EXPECT_EQ(dut.finger_face_contacts()[4], finger_face_contacts_expected);
  EXPECT_EQ(dut.finger_face_contacts()[5], finger_face_contacts_expected);
  finger_face_contacts_expected.erase(
      finger_face_contacts_expected.find(Finger::kFinger2));
  EXPECT_EQ(dut.finger_face_contacts()[6], finger_face_contacts_expected);
  finger_face_contacts_expected.emplace(Finger::kFinger2, BrickFace::kPosY);
  EXPECT_EQ(dut.finger_face_contacts()[7], finger_face_contacts_expected);
  EXPECT_EQ(dut.finger_face_contacts()[8], finger_face_contacts_expected);

  dut.get_mutable_prog()->AddBoundingBoxConstraint(0.1, 0.4, dut.dt());

  // Finger don't move too fast.
  dut.AddPositionDifferenceBound(
      1, gripper_brick.finger_base_position_index(Finger::kFinger1),
      0.2 * M_PI);
  dut.AddPositionDifferenceBound(
      1, gripper_brick.finger_mid_position_index(Finger::kFinger1), 0.2 * M_PI);
  dut.AddPositionDifferenceBound(
      2, gripper_brick.finger_base_position_index(Finger::kFinger1),
      0.2 * M_PI);
  dut.AddPositionDifferenceBound(
      2, gripper_brick.finger_mid_position_index(Finger::kFinger1), 0.2 * M_PI);
  dut.AddPositionDifferenceBound(
      5, gripper_brick.finger_base_position_index(Finger::kFinger2),
      0.2 * M_PI);
  dut.AddPositionDifferenceBound(
      5, gripper_brick.finger_mid_position_index(Finger::kFinger2), 0.1 * M_PI);
  dut.AddPositionDifferenceBound(
      6, gripper_brick.finger_base_position_index(Finger::kFinger2),
      0.2 * M_PI);
  dut.AddPositionDifferenceBound(
      6, gripper_brick.finger_mid_position_index(Finger::kFinger2), 0.1 * M_PI);

  // Constraint that the brick is rotated by at least certain angle.
  dut.get_mutable_prog()->AddLinearConstraint(
      dut.q_vars()(gripper_brick.brick_revolute_x_position_index(), nT - 1) -
          dut.q_vars()(gripper_brick.brick_revolute_x_position_index(), 0),
      -kInf, -0.1 * M_PI);

  Eigen::VectorXd x_guess(dut.prog().num_vars());
  x_guess.setZero();
  Eigen::VectorXd q0_guess(gripper_brick.plant().num_positions());
  q0_guess << -0.3, 0.9, 0.3, 0, 0, -1.2, 0.2, 0, -0.4;
  dut.prog().SetDecisionVariableValueInVector(dut.q_vars().col(0), q0_guess,
                                              &x_guess);

  solvers::SnoptSolver snopt_solver;
  if (snopt_solver.available()) {
    const auto result = snopt_solver.Solve(dut.prog(), x_guess, {});
    std::cout << "snopt info: "
              << result.get_solver_details<solvers::SnoptSolver>().info << "\n";
    ASSERT_TRUE(result.is_success());

    // Check the position interpolation constraint
    const Eigen::MatrixXd q_sol = result.GetSolution(dut.q_vars());
    const Eigen::VectorXd brick_v_y_sol =
        result.GetSolution(dut.brick_v_y_vars());
    const Eigen::VectorXd brick_v_z_sol =
        result.GetSolution(dut.brick_v_z_vars());
    const Eigen::VectorXd brick_omega_x_sol =
        result.GetSolution(dut.brick_omega_x_vars());
    const Eigen::VectorXd dt_sol = result.GetSolution(dut.dt());
    // Check the integration constraint on brick pose.
    for (int i = 0; i < nT - 1; ++i) {
      EXPECT_NEAR(
          q_sol(gripper_brick.brick_translate_y_position_index(), i + 1) -
              q_sol(gripper_brick.brick_translate_y_position_index(), i),
          (brick_v_y_sol(i) + brick_v_y_sol(i + 1)) / 2 * dt_sol(i), 1E-6);
      EXPECT_NEAR(
          q_sol(gripper_brick.brick_translate_z_position_index(), i + 1) -
              q_sol(gripper_brick.brick_translate_z_position_index(), i),
          (brick_v_z_sol(i) + brick_v_z_sol(i + 1)) / 2 * dt_sol(i), 1E-6);
      EXPECT_NEAR(
          q_sol(gripper_brick.brick_revolute_x_position_index(), i + 1) -
              q_sol(gripper_brick.brick_revolute_x_position_index(), i),
          (brick_omega_x_sol(i) + brick_omega_x_sol(i + 1)) / 2 * dt_sol(i),
          1E-6);
    }

    // Now check the integration constraint on the brick velocity/acceleration.
    auto check_dynamics =
        [&result, &dut, &gripper_brick, brick_lid_friction_force_magnitude,
         brick_lid_friction_torque_magnitude, &q_sol, &brick_v_y_sol,
         &brick_v_z_sol, &brick_omega_x_sol,
         &dt_sol](int left_knot_index,
                  const std::map<Finger, BrickFace>& finger_face_contacts) {
          BrickDynamicBackwardEulerConstraint constraint(
              &gripper_brick, dut.plant_mutable_context(left_knot_index + 1),
              finger_face_contacts, brick_lid_friction_force_magnitude,
              brick_lid_friction_torque_magnitude);
          const int right_knot_index = left_knot_index + 1;
          Eigen::Matrix2Xd f_FB_B_r_sol(2, finger_face_contacts.size());
          int f_FB_B_r_count = 0;
          for (const auto& finger_face_contact : finger_face_contacts) {
            f_FB_B_r_sol.col(f_FB_B_r_count++) = result.GetSolution(
                dut.f_FB_B()[right_knot_index].at(finger_face_contact.first));
          }
          Eigen::VectorXd constraint_x;
          constraint.ComposeX<double>(
              q_sol.col(right_knot_index), brick_v_y_sol(right_knot_index),
              brick_v_z_sol(right_knot_index),
              brick_omega_x_sol(right_knot_index),
              brick_v_y_sol(left_knot_index), brick_v_z_sol(left_knot_index),
              brick_omega_x_sol(left_knot_index), f_FB_B_r_sol,
              dt_sol(left_knot_index), &constraint_x);
          EXPECT_TRUE(constraint.CheckSatisfied(constraint_x, 1E-6));
        };
    for (int i = 0; i < nT - 1; ++i) {
      if (i == 1 || i == 5) {
        check_dynamics(i, dut.finger_face_contacts()[i + 1]);
      } else {
        check_dynamics(i, dut.finger_face_contacts()[i]);
      }
    }

    auto diagram_context = gripper_brick.diagram().CreateDefaultContext();
    systems::Context<double>* plant_mutable_context =
        &(gripper_brick.diagram().GetMutableSubsystemContext(
            gripper_brick.plant(), diagram_context.get()));
    for (int i = 0; i < nT; ++i) {
      gripper_brick.plant().SetPositions(plant_mutable_context, q_sol.col(i));
      gripper_brick.diagram().Publish(*diagram_context);
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  }
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
