#include <memory>

#include <limits>

#include "drake/examples/planar_gripper/gripper_brick.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
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
  Eigen::Vector3d p_BTip_lower = -box_size * 0.45;
  Eigen::Vector3d p_BTip_upper = box_size * 0.45;
  const double finger_tip_radius = 0.015;
  switch (brick_face) {
    case BrickFace::kPosZ: {
      p_BTip_lower(2) = box_size(2) / 2 + finger_tip_radius;
      p_BTip_upper(2) = box_size(2) / 2 + finger_tip_radius;
      break;
    }
    case BrickFace::kNegZ: {
      p_BTip_lower(2) = -box_size(2) / 2 - finger_tip_radius;
      p_BTip_upper(2) = -box_size(2) / 2 - finger_tip_radius;
      break;
    }
    case BrickFace::kPosY: {
      p_BTip_lower(1) = box_size(1) / 2 + finger_tip_radius;
      p_BTip_upper(1) = box_size(1) / 2 + finger_tip_radius;
      break;
    }
    case BrickFace::kNegY: {
      p_BTip_lower(1) = -box_size(1) / 2 - finger_tip_radius;
      p_BTip_upper(1) = -box_size(1) / 2 - finger_tip_radius;
      break;
    }
  }
  ik->AddPositionConstraint(finger_link2, p_F2Tip, brick, p_BTip_lower,
                            p_BTip_upper);
}

void FixFingerPositionInBrickFrame(
    const multibody::MultibodyPlant<double>& plant,
    const systems::Context<double>& fixed_context, int finger_index,
    multibody::InverseKinematics* ik) {
  const Eigen::Vector3d p_F2Tip(0, 0, -0.086);
  const multibody::Frame<double>& finger_link2 =
      plant.GetFrameByName("finger" + std::to_string(finger_index) + "_link2");
  const multibody::Frame<double>& brick = plant.GetFrameByName("brick_link");
  Eigen::Vector3d p_BTip;
  plant.CalcPointsPositions(fixed_context, finger_link2, p_F2Tip, brick,
                            &p_BTip);
  ik->AddPositionConstraint(finger_link2, p_F2Tip, brick, p_BTip, p_BTip);
}

Eigen::VectorXd FindInitialPosture(
    const GripperBrickSystem<double>& gripper_brick_system,
    systems::Context<double>* plant_mutable_context,
    systems::Context<double>* diagram_context) {
  multibody::InverseKinematics ik0(gripper_brick_system.plant(),
                                   plant_mutable_context);

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

Eigen::VectorXd FindPosture2(
    const GripperBrickSystem<double>& gripper_brick_system,
    systems::Context<double>* plant_mutable_context,
    systems::Context<double>* diagram_context) {
  multibody::InverseKinematics ik(gripper_brick_system.plant(),
                                  plant_mutable_context);

  FixFingerPositionInBrickFrame(gripper_brick_system.plant(),
                                *plant_mutable_context, 1, &ik);
  FixFingerPositionInBrickFrame(gripper_brick_system.plant(),
                                *plant_mutable_context, 2, &ik);
  AddFingerTipInContactWithBrickFace(gripper_brick_system.plant(), 3,
                                     BrickFace::kNegZ, &ik);

  auto result = solvers::Solve(
      ik.prog(),
      gripper_brick_system.plant().GetPositions(*plant_mutable_context));
  std::cout << result.get_solution_result() << "\n";

  const Eigen::VectorXd q = result.GetSolution(ik.q());
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q);

  gripper_brick_system.diagram().Publish(*diagram_context);
  return q;
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

void InterpolateTrajectory(const multibody::MultibodyPlant<double>& plant,
                           int interpolation_level,
                           const optional<int>& moving_finger_index,
                           std::list<Eigen::VectorXd>* q_samples,
                           systems::Context<double>* plant_mutable_context) {
  if (interpolation_level == 0) {
    return;
  }
  auto it1 = q_samples->begin();
  auto it2 = (++it1);
  --it1;
  while (it2 != q_samples->end()) {
    plant.SetPositions(plant_mutable_context, q_samples->front());
    multibody::InverseKinematics ik(plant, plant_mutable_context);
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

    const Eigen::VectorXd q_guess = (*it1 + *it2) / 2;
    ik.get_mutable_prog()->AddQuadraticErrorCost(
        Eigen::MatrixXd::Identity(plant.num_positions(), plant.num_positions()),
        q_guess, ik.q());
    ik.get_mutable_prog()->AddBoundingBoxConstraint(
        q_guess - 0.1 * Eigen::VectorXd::Ones(plant.num_positions()),
        q_guess + 0.1 * Eigen::VectorXd::Ones(plant.num_positions()), ik.q());
    const auto result = solvers::Solve(ik.prog(), q_guess);
    q_samples->insert(it2, result.GetSolution(ik.q()));
    it1 = it2;
    it2++;
  }
  InterpolateTrajectory(plant, interpolation_level - 1, moving_finger_index,
                        q_samples, plant_mutable_context);
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

int DoMain() {
  GripperBrickSystem<double> gripper_brick_system(true /* add gravity */);

  // const int brick_joint_index = gripper_brick_system.plant()
  //                                  .GetJointByName("box_pin_joint")
  //                                  .position_start();

  auto diagram_context = gripper_brick_system.diagram().CreateDefaultContext();
  systems::Context<double>* plant_mutable_context =
      &(gripper_brick_system.diagram().GetMutableSubsystemContext(
          gripper_brick_system.plant(), diagram_context.get()));

  // Find the initial posture that all three fingers are in contact.
  const Eigen::VectorXd q0 = FindInitialPosture(
      gripper_brick_system, plant_mutable_context, diagram_context.get());

  do {
    std::cout << "Type y to continue.\n";
  } while (std::cin.get() != 'y');
  // Now rotate the brick by certain degrees.
  multibody::InverseKinematics ik1(gripper_brick_system.plant(),
                                   plant_mutable_context);
  RotateBoxByCertainDegree(gripper_brick_system.plant(), 50.0 / 180 * M_PI,
                           60.0 / 180 * M_PI, *plant_mutable_context, &ik1);
  const solvers::MathematicalProgramResult result =
      solvers::Solve(ik1.prog(), q0);
  std::cout << result.get_solution_result() << "\n";
  const Eigen::VectorXd q1 = result.GetSolution(ik1.q());

  gripper_brick_system.plant().SetPositions(plant_mutable_context, q1);

  gripper_brick_system.diagram().Publish(*diagram_context);

  // Move finger 3 to negZ face.
  do {
    std::cout << "Type y to continue.\n";
  } while (std::cin.get() != 'y');
  const Eigen::VectorXd q2 = FindPosture2(
      gripper_brick_system, plant_mutable_context, diagram_context.get());

  std::list<Eigen::VectorXd> q_move2 = {q1, q2};
  InterpolateTrajectory(gripper_brick_system.plant(), 4, 3, &q_move2,
                        plant_mutable_context);
  for (const auto& q_move2_sample : q_move2) {
    VisualizePosture(gripper_brick_system, q_move2_sample,
                     plant_mutable_context, diagram_context.get());
    sleep(1);
  }

  return 0;
}
}  // namespace examples
}  // namespace drake

int main() { drake::examples::DoMain(); }
