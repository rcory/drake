#include "drake/examples/planar_gripper/gripper_brick_trajectory_optimization.h"
#include "drake/solvers/snopt_solver.h"

namespace drake {
namespace examples {
namespace planar_gripper {
int DoMain() {
  GripperBrickHelper<double> gripper_brick;
  int nT = 16;
  std::map<Finger, BrickFace> initial_contact(
      {{Finger::kFinger1, BrickFace::kNegY},
       {Finger::kFinger2, BrickFace::kNegZ},
       {Finger::kFinger3, BrickFace::kPosY}});

  std::vector<FingerTransition> finger_transitions;
  finger_transitions.emplace_back(1, 4, Finger::kFinger1, BrickFace::kPosZ);
  finger_transitions.emplace_back(6, 9, Finger::kFinger3, BrickFace::kNegZ);
  finger_transitions.emplace_back(11, 14, Finger::kFinger2, BrickFace::kNegY);

  const double brick_lid_friction_force_magnitude = 0;
  const double brick_lid_friction_torque_magnitude = 0;

  GripperBrickTrajectoryOptimization dut(
      &gripper_brick, nT, initial_contact, finger_transitions,
      brick_lid_friction_force_magnitude, brick_lid_friction_torque_magnitude,
      GripperBrickTrajectoryOptimization::Options(
          0.8,
          GripperBrickTrajectoryOptimization::IntegrationMethod::kBackwardEuler,
          0.05 * M_PI, 0.03));

  dut.get_mutable_prog()->AddBoundingBoxConstraint(0.1, 0.4, dut.dt());

  // Initial pose constraint on the brick.
  dut.get_mutable_prog()->AddBoundingBoxConstraint(
      0, 0, dut.q_vars()(gripper_brick.brick_revolute_x_position_index(), 0));

  // Final pose constraint on the brick.
  dut.get_mutable_prog()->AddBoundingBoxConstraint(
      0.3 * M_PI, 0.4 * M_PI,
      dut.q_vars()(gripper_brick.brick_revolute_x_position_index(), nT - 1));

  // Fingers cannot move too fast
  for (const auto& finger_transition : finger_transitions) {
    for (int i = finger_transition.start_knot_index;
         i < finger_transition.end_knot_index; ++i) {
      dut.AddPositionDifferenceBound(
          i, gripper_brick.finger_base_position_index(finger_transition.finger),
          0.1 * M_PI);
      dut.AddPositionDifferenceBound(
          i, gripper_brick.finger_mid_position_index(finger_transition.finger),
          0.1 * M_PI);
    }
  }

  dut.AddBrickStaticEquilibriumConstraint(0);
  dut.AddBrickStaticEquilibriumConstraint(nT - 1);
  dut.get_mutable_prog()->SetSolverOption(solvers::SnoptSolver::id(),
                                          "iterations limit", 100000);

  // dut.AddCollisionAvoidanceForInterpolatedPosture(
  //    finger_transitions[0].end_knot_index - 1, 0.9,
  //    {gripper_brick.brick_geometry_id(),
  //     gripper_brick.finger_tip_sphere_geometry_id(
  //         finger_transitions[0].finger)},
  //    0.01);
  // dut.AddCollisionAvoidanceForInterpolatedPosture(
  //    finger_transitions[0].end_knot_index - 1, 0.8,
  //    {gripper_brick.brick_geometry_id(),
  //     gripper_brick.finger_tip_sphere_geometry_id(
  //         finger_transitions[0].finger)},
  //    0.02);
  // dut.AddCollisionAvoidanceForInterpolatedPosture(
  //    finger_transitions[0].end_knot_index - 1, 0.7,
  //    {gripper_brick.brick_geometry_id(),
  //     gripper_brick.finger_tip_sphere_geometry_id(
  //         finger_transitions[0].finger)},
  //    0.02);
  // dut.AddCollisionAvoidanceForInterpolatedPosture(
  //    finger_transitions[2].end_knot_index - 1, 0.7,
  //    {gripper_brick.brick_geometry_id(),
  //     gripper_brick.finger_tip_sphere_geometry_id(
  //         finger_transitions[2].finger)},
  //    0.02);

  solvers::SnoptSolver solver;
  if (solver.is_available()) {
    const solvers::MathematicalProgramResult result =
        solver.Solve(dut.prog(), {}, {});
    std::cout << "info: "
              << result.get_solver_details<solvers::SnoptSolver>().info << "\n";
    std::cout << result.get_solution_result() << "\n";

    const Eigen::MatrixXd q_sol = result.GetSolution(dut.q_vars());
    const Eigen::VectorXd dt_sol = result.GetSolution(dut.dt());
    std::cout << "dt: " << dt_sol.transpose() << "\n";

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
  return 0;
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

int main() { return drake::examples::planar_gripper::DoMain(); }
