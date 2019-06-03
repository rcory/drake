#include <memory>

#include <iomanip>

#include <limits>

#include "drake/examples/planar_gripper/gripper_brick.h"
#include "drake/multibody/inverse_kinematics/distance_constraint.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/multibody/inverse_kinematics/position_constraint.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace examples {

const double kInf = std::numeric_limits<double>::infinity();

void VisualizePosture(const GripperBrickSystem<double>& gripper_brick_system,
                      const Eigen::Ref<const Eigen::VectorXd>& q,
                      systems::Context<double>* plant_mutable_context,
                      systems::Context<double>* diagram_context) {
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q);

  gripper_brick_system.diagram().Publish(*diagram_context);
}

void AddFingerTipInContactWithBrickFace(
    const multibody::MultibodyPlant<double>& plant, int finger_index,
    BrickFace brick_face, multibody::InverseKinematics* ik) {
  const multibody::Frame<double>& finger_link2 =
      plant.GetFrameByName("finger" + std::to_string(finger_index) + "_link2");
  // position of Tip in the finger link 2 farme (F2).
  const Eigen::Vector3d p_F2Tip(0, 0, -0.086);
  const multibody::Frame<double>& brick = plant.GetFrameByName("brick_link");
  const Eigen::Vector3d box_size(0.025, 0.077, 0.077);
  Eigen::Vector3d p_BTip_lower = -box_size * 0.42;
  Eigen::Vector3d p_BTip_upper = box_size * 0.42;
  const double finger_tip_radius = 0.015;
  const double depth = 1E-3;
  switch (brick_face) {
    case BrickFace::kPosZ: {
      p_BTip_lower(2) = box_size(2) / 2 + finger_tip_radius - depth;
      p_BTip_upper(2) = box_size(2) / 2 + finger_tip_radius - depth;
      break;
    }
    case BrickFace::kNegZ: {
      p_BTip_lower(2) = -box_size(2) / 2 - finger_tip_radius + depth;
      p_BTip_upper(2) = -box_size(2) / 2 - finger_tip_radius + depth;
      break;
    }
    case BrickFace::kPosY: {
      p_BTip_lower(1) = box_size(1) / 2 + finger_tip_radius - depth;
      p_BTip_upper(1) = box_size(1) / 2 + finger_tip_radius - depth;
      break;
    }
    case BrickFace::kNegY: {
      p_BTip_lower(1) = -box_size(1) / 2 - finger_tip_radius + depth;
      p_BTip_upper(1) = -box_size(1) / 2 - finger_tip_radius + depth;
      break;
    }
  }
  ik->AddPositionConstraint(finger_link2, p_F2Tip, brick, p_BTip_lower,
                            p_BTip_upper);
}

void FixFingerPositionInBrickFrame(
    const multibody::MultibodyPlant<double>& plant,
    const systems::Context<double>& fixed_context, int finger_index,
    solvers::MathematicalProgram* prog,
    systems::Context<double>* plant_mutable_context,
    const VectorX<symbolic::Variable>& q) {
  const Eigen::Vector3d p_F2Tip(0, 0, -0.086);
  const multibody::Frame<double>& finger_link2 =
      plant.GetFrameByName("finger" + std::to_string(finger_index) + "_link2");
  const multibody::Frame<double>& brick = plant.GetFrameByName("brick_link");
  Eigen::Vector3d p_BTip;
  plant.CalcPointsPositions(fixed_context, finger_link2, p_F2Tip, brick,
                            &p_BTip);
  prog->AddConstraint(std::make_shared<multibody::PositionConstraint>(
                          &plant, brick, p_BTip, p_BTip, finger_link2, p_F2Tip,
                          plant_mutable_context),
                      q);
}

void FixFingerPositionInBrickFrame(
    const multibody::MultibodyPlant<double>& plant,
    const systems::Context<double>& fixed_context, int finger_index,
    multibody::InverseKinematics* ik) {
  FixFingerPositionInBrickFrame(plant, fixed_context, finger_index,
                                ik->get_mutable_prog(),
                                ik->get_mutable_context(), ik->q());
}

Eigen::MatrixXd FindTrajectory(
    const multibody::MultibodyPlant<double>& plant,
    const Eigen::Ref<const Eigen::VectorXd>& q_start,
    const Eigen::Ref<const Eigen::VectorXd>& q_end, int num_samples,
    const optional<int>& moving_finger_index,
    systems::Context<double>* plant_mutable_context) {
  DRAKE_ASSERT(num_samples >= 3);
  Eigen::MatrixXd q_guess(plant.num_positions(), num_samples);
  Eigen::MatrixXd q_sol(plant.num_positions(), num_samples);
  q_sol.col(0) = q_start;
  q_sol.col(num_samples - 1) = q_end;
  for (int i = 0; i < plant.num_positions(); ++i) {
    q_guess.row(i) =
        Eigen::RowVectorXd::LinSpaced(num_samples, q_start(i), q_end(i));
  }

  DRAKE_DEMAND(
      plant.GetCollisionGeometriesForBody(plant.GetBodyByName("brick_link"))
          .size() == 1);
  DRAKE_DEMAND(plant
                   .GetCollisionGeometriesForBody(plant.GetBodyByName(
                       "finger" + std::to_string(moving_finger_index.value()) +
                       "_link2"))
                   .size() == 1);
  for (int i = 1; i < num_samples - 1; ++i) {
    multibody::InverseKinematics ik(plant, plant_mutable_context);
    plant.SetPositions(plant_mutable_context, q_start);
    for (int j = 1; j <= 3; ++j) {
      if (moving_finger_index.has_value() && moving_finger_index.value() == j) {
        SortedPair<geometry::GeometryId> geometry_pair(
            plant.GetCollisionGeometriesForBody(
                plant.GetBodyByName("brick_link"))[0],
            plant.GetCollisionGeometriesForBody(plant.GetBodyByName(
                "finger" + std::to_string(moving_finger_index.value()) +
                "_link2"))[0]);
        ik.AddDistanceConstraint(geometry_pair, 0.01, kInf);
      } else {
        FixFingerPositionInBrickFrame(plant, *plant_mutable_context, j, &ik);
      }
    }

    ik.get_mutable_prog()->AddQuadraticErrorCost(
        Eigen::MatrixXd::Identity(plant.num_positions(), plant.num_positions()),
        q_guess.col(i), ik.q());
    const auto result = solvers::Solve(ik.prog(), q_guess.col(i));
    q_sol.col(i) = result.GetSolution(ik.q());
  }
  return q_sol;
}

Eigen::MatrixXd InterpolateTrajectory(
    const systems::Diagram<double>& diagram,
    const multibody::MultibodyPlant<double>& plant, int num_samples,
    const optional<int>& moving_finger_index,
    const Eigen::Ref<const Eigen::VectorXd>& q_start,
    const Eigen::Ref<const Eigen::VectorXd>& q_end,
    const systems::Context<double>& plant_fixed_context,
    const Eigen::Ref<const Eigen::VectorXd>& delta_q_max) {
  solvers::MathematicalProgram prog;
  auto q = prog.NewContinuousVariables(plant.num_positions(), num_samples);
  std::vector<std::unique_ptr<systems::Context<double>>> diagram_contexts;
  std::vector<systems::Context<double>*> plant_contexts;
  for (int i = 0; i < num_samples; ++i) {
    auto diagram_context = diagram.CreateDefaultContext();
    diagram_contexts.push_back(std::move(diagram_context));
    plant_contexts.push_back(&(diagram.GetMutableSubsystemContext(
        plant, diagram_contexts.back().get())));
  }
  prog.AddBoundingBoxConstraint(q_start, q_start, q.col(0));
  prog.AddBoundingBoxConstraint(q_end, q_end, q.col(num_samples - 1));
  std::vector<std::unique_ptr<systems::Context<double>>>
      diagram_contexts_midpoint;
  std::vector<systems::Context<double>*> plant_mutable_contexts_midpoint;
  auto q_middle =
      prog.NewContinuousVariables(plant.num_positions(), num_samples - 1);

  for (int i = 0; i < num_samples - 1; ++i) {
    auto diagram_context = diagram.CreateDefaultContext();
    diagram_contexts_midpoint.push_back(std::move(diagram_context));
    plant_mutable_contexts_midpoint.push_back(
        &(diagram.GetMutableSubsystemContext(
            plant, diagram_contexts_midpoint.back().get())));
    prog.AddLinearEqualityConstraint(
        q.col(i) + q.col(i + 1) - 2.0 * q_middle.col(i),
        Eigen::VectorXd::Zero(plant.num_positions()));
    prog.AddLinearConstraint(q.col(i + 1) - q.col(i), -delta_q_max,
                             delta_q_max);
  }
  for (int i = 0; i < num_samples - 1; ++i) {
    for (int j = 1; j <= 3; ++j) {
      if (moving_finger_index.has_value() && moving_finger_index.value() == j) {
        SortedPair<geometry::GeometryId> geometry_pair(
            plant.GetCollisionGeometriesForBody(
                plant.GetBodyByName("brick_link"))[0],
            plant.GetCollisionGeometriesForBody(plant.GetBodyByName(
                "finger" + std::to_string(moving_finger_index.value()) +
                "_link2"))[0]);
        if (i >= 1) {
          prog.AddConstraint(
              std::make_shared<multibody::DistanceConstraint>(
                  &plant, geometry_pair, plant_contexts[i], 0.01, kInf),
              q.col(i));
        }
        prog.AddConstraint(std::make_shared<multibody::DistanceConstraint>(
                               &plant, geometry_pair,
                               plant_mutable_contexts_midpoint[i], 0.01, kInf),
                           q_middle.col(i));
      } else {
        if (i >= 1) {
          FixFingerPositionInBrickFrame(plant, plant_fixed_context, j, &prog,
                                        plant_contexts[i], q.col(i));
        }
      }
    }
  }

  // Add the constraint that the mid point between two samples is also collision
  // free.

  for (int i = 0; i < num_samples - 1; ++i) {
    prog.AddQuadraticCost((q.col(i + 1) - q.col(i)).squaredNorm());
  }
  Eigen::MatrixXd q_guess(plant.num_positions(), num_samples);
  for (int i = 0; i < plant.num_positions(); ++i) {
    q_guess.row(i) =
        Eigen::RowVectorXd::LinSpaced(num_samples, q_start(i), q_end(i));
  }
  Eigen::VectorXd q_guess_stacked(plant.num_positions() * num_samples);
  for (int i = 0; i < num_samples; ++i) {
    q_guess_stacked.block(i * plant.num_positions(), 0, plant.num_positions(),
                          1) = q_guess.col(i);
  }
  const auto result = solvers::Solve(prog, q_guess_stacked);
  std::cout << "interpolate trajectory " << result.get_solution_result()
            << "\n";
  return result.GetSolution(q);
}

void RotateBoxByCertainDegree(const multibody::MultibodyPlant<double>& plant,
                              double rotate_angle_lower,
                              double rotate_angle_upper,
                              const systems::Context<double>& plant_context,
                              multibody::InverseKinematics* ik) {
  for (int i = 1; i <= 3; ++i) {
    FixFingerPositionInBrickFrame(plant, plant_context, i, ik);
  }
  const int brick_theta_index =
      plant.GetJointByName("brick_revolute_x_joint").position_start();
  const Eigen::VectorXd q0 = plant.GetPositions(plant_context);
  ik->get_mutable_prog()->AddBoundingBoxConstraint(
      q0(brick_theta_index) + rotate_angle_lower,
      q0(brick_theta_index) + rotate_angle_upper, ik->q()(brick_theta_index));
}
Eigen::VectorXd FindInitialPosture(
    const GripperBrickSystem<double>& gripper_brick_system,
    systems::Context<double>* plant_mutable_context,
    systems::Context<double>* diagram_context) {
  multibody::InverseKinematics ik0(gripper_brick_system.plant(),
                                   plant_mutable_context);

  // Finger 1 in +Z face. Finger 2 in -Z face. Finger 3 in +Y face.
  AddFingerTipInContactWithBrickFace(gripper_brick_system.plant(), 1,
                                     BrickFace::kPosZ, &ik0);
  AddFingerTipInContactWithBrickFace(gripper_brick_system.plant(), 2,
                                     BrickFace::kNegZ, &ik0);
  AddFingerTipInContactWithBrickFace(gripper_brick_system.plant(), 3,
                                     BrickFace::kPosY, &ik0);

  auto result = solvers::Solve(ik0.prog());
  std::cout << result.get_solution_result() << "\n";

  const Eigen::VectorXd q0 = result.GetSolution(ik0.q());
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q0);

  gripper_brick_system.diagram().Publish(*diagram_context);
  return q0;
}

Eigen::VectorXd FindPosture(
    const GripperBrickSystem<double>& gripper_brick_system,
    const std::vector<int>& fixed_fingers,
    const std::vector<std::pair<int, BrickFace>>& moving_fingers,
    systems::Context<double>* plant_mutable_context) {
  multibody::InverseKinematics ik(gripper_brick_system.plant(),
                                  plant_mutable_context);
  for (int fixed_finger : fixed_fingers) {
    FixFingerPositionInBrickFrame(gripper_brick_system.plant(),
                                  *plant_mutable_context, fixed_finger, &ik);
  }
  for (const auto& moving_finger : moving_fingers) {
    AddFingerTipInContactWithBrickFace(gripper_brick_system.plant(),
                                       moving_finger.first,
                                       moving_finger.second, &ik);
  }
  auto result = solvers::Solve(
      ik.prog(),
      gripper_brick_system.plant().GetPositions(*plant_mutable_context));
  std::cout << result.get_solution_result() << "\n";

  const Eigen::VectorXd q = result.GetSolution(ik.q());
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q);

  return q;
}

Eigen::VectorXd FindPosture1(
    const GripperBrickSystem<double>& gripper_brick_system,
    systems::Context<double>* plant_mutable_context,
    systems::Context<double>* diagram_context) {
  unused(diagram_context);
  const Eigen::VectorXd q1 =
      FindPosture(gripper_brick_system, {2, 3}, {{1, BrickFace::kNegY}},
                  plant_mutable_context);
  return q1;
}

Eigen::MatrixXd RotateBlockToPosture(
    const GripperBrickSystem<double>& gripper_brick_system,
    const Eigen::Ref<const Eigen::VectorXd>& q1, double angle_lower,
    double angle_upper, const Eigen::Ref<const Eigen::VectorXd>& delta_q_max,
    int num_samples, systems::Context<double>* plant_mutable_context) {
  multibody::InverseKinematics ik2(gripper_brick_system.plant(),
                                   plant_mutable_context);
  RotateBoxByCertainDegree(gripper_brick_system.plant(), angle_lower,
                           angle_upper, *plant_mutable_context, &ik2);
  const solvers::MathematicalProgramResult result =
      solvers::Solve(ik2.prog(), q1);
  std::cout << result.get_solution_result() << "\n";
  const Eigen::VectorXd q2 = result.GetSolution(ik2.q());

  gripper_brick_system.plant().SetPositions(plant_mutable_context, q1);

  // Now find the samples from q1 to q2
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q1);
  const Eigen::MatrixXd q_move = InterpolateTrajectory(
      gripper_brick_system.diagram(), gripper_brick_system.plant(), num_samples,
      {}, q1, q2, *plant_mutable_context, delta_q_max);
  return q_move;
}

Eigen::MatrixXd RotateBlockToPosture2(
    const GripperBrickSystem<double>& gripper_brick_system,
    const Eigen::Ref<const Eigen::VectorXd>& q1,
    systems::Context<double>* plant_mutable_context) {
  Eigen::VectorXd delta_q_max = 0.3 * Eigen::VectorXd::Ones(q1.rows());
  return RotateBlockToPosture(gripper_brick_system, q1, 60.0 / 180 * M_PI,
                              70.0 / 180 * M_PI, delta_q_max, 7, 
                              plant_mutable_context);
}

Eigen::VectorXd FindPosture2(
    const GripperBrickSystem<double>& gripper_brick_system,
    systems::Context<double>* plant_mutable_context,
    systems::Context<double>* diagram_context) {
  unused(diagram_context);
  const Eigen::VectorXd q2 =
      FindPosture(gripper_brick_system, {1, 2}, {{3, BrickFace::kNegZ}},
                  plant_mutable_context);
  return q2;
}

int DoMain() {
  GripperBrickSystem<double> gripper_brick_system;

  auto diagram_context = gripper_brick_system.diagram().CreateDefaultContext();
  systems::Context<double>* plant_mutable_context =
      &(gripper_brick_system.diagram().GetMutableSubsystemContext(
          gripper_brick_system.plant(), diagram_context.get()));

  // Find the initial posture that all three fingers are in contact.
  const Eigen::VectorXd q0 = FindInitialPosture(
      gripper_brick_system, plant_mutable_context, diagram_context.get());
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q0);

  do {
    std::cout << "Type y to continue.\n";
  } while (std::cin.get() != 'y');
  // Now move finger 1 to negY face.
  const Eigen::VectorXd q1 = FindPosture1(
      gripper_brick_system, plant_mutable_context, diagram_context.get());
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q1);
  Eigen::VectorXd delta_q_max =
      0.1 * Eigen::VectorXd::Ones(gripper_brick_system.plant().num_positions());
  const Eigen::MatrixXd q_move1 = InterpolateTrajectory(
      gripper_brick_system.diagram(), gripper_brick_system.plant(), 7, 1, q0,
      q1, *plant_mutable_context, delta_q_max);
  for (int i = 0; i < q_move1.cols(); ++i) {
    VisualizePosture(gripper_brick_system, q_move1.col(i),
                     plant_mutable_context, diagram_context.get());
    sleep(1);
  }
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q1);

  // Now rotate the brick by certain degrees.
  const Eigen::MatrixXd q_move2 =
      RotateBlockToPosture2(gripper_brick_system, q1, plant_mutable_context);
  for (int i = 0; i < q_move2.cols(); ++i) {
    VisualizePosture(gripper_brick_system, q_move2.col(i),
                     plant_mutable_context, diagram_context.get());
    sleep(1);
  }

  // Move finger 3 to negZ face.
  const Eigen::VectorXd q3 = FindPosture2(
      gripper_brick_system, plant_mutable_context, diagram_context.get());

  gripper_brick_system.plant().SetPositions(plant_mutable_context, q3);
  const Eigen::MatrixXd q_move3 = InterpolateTrajectory(
      gripper_brick_system.diagram(), gripper_brick_system.plant(), 8, 3,
      q_move2.col(q_move2.cols() - 1), q3, *plant_mutable_context, delta_q_max);
  for (int i = 0; i < q_move3.cols(); ++i) {
    VisualizePosture(gripper_brick_system, q_move3.col(i),
                     plant_mutable_context, diagram_context.get());
    sleep(1);
  }

  // Move finger 2 to negY face.
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q3);
  const Eigen::VectorXd q4 =
      FindPosture(gripper_brick_system, {1, 3}, {{2, BrickFace::kNegY}},
                  plant_mutable_context);
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q4);
  const Eigen::MatrixXd q_move4 = InterpolateTrajectory(
      gripper_brick_system.diagram(), gripper_brick_system.plant(), 8, 2, q3,
      q4, *plant_mutable_context, delta_q_max);
  for (int i = 0; i < q_move4.cols(); ++i) {
    VisualizePosture(gripper_brick_system, q_move4.col(i),
                     plant_mutable_context, diagram_context.get());
    sleep(1);
  }

  // Move finger 1 to posZ face
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q4);
  const Eigen::VectorXd q5 =
      FindPosture(gripper_brick_system, {2, 3}, {{1, BrickFace::kPosZ}},
                  plant_mutable_context);
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q5);
  const Eigen::MatrixXd q_move5 = InterpolateTrajectory(
      gripper_brick_system.diagram(), gripper_brick_system.plant(), 9, 1, q4,
      q5, *plant_mutable_context, delta_q_max);
  for (int i = 0; i < q_move5.cols(); ++i) {
    VisualizePosture(gripper_brick_system, q_move5.col(i),
                     plant_mutable_context, diagram_context.get());
    sleep(1);
  }

  // Now rotate the brick by certain degrees.
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q5);
  const Eigen::MatrixXd q_move6 = RotateBlockToPosture(
      gripper_brick_system, q5, 40.0 / 180 * M_PI, 50.0 / 180 * M_PI,
      delta_q_max, 9, plant_mutable_context);
  for (int i = 0; i < q_move6.cols(); ++i) {
    VisualizePosture(gripper_brick_system, q_move6.col(i),
                     plant_mutable_context, diagram_context.get());
    sleep(1);
  }

  // std::stringstream ss;
  // ss << std::setprecision(20);
  // const int brick_joint_index = gripper_brick_system.plant()
  //                                  .GetJointByName("brick_translate_y_joint")
  //                                  .position_start();

  // ss << q0.segment<3>(brick_joint_index) << "\n";
  // ss << q_move1.topRows<6>().transpose() << "\n";
  // ss << q_move2.topRows<6>().transpose() << "\n";
  // std::cout << ss.str() << "\n";

  return 0;
}
}  // namespace examples
}  // namespace drake

int main() { drake::examples::DoMain(); }
