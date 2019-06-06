#include <memory>

#include <iomanip>

#include <limits>

#include "drake/examples/planar_gripper/gripper_brick.h"
#include "drake/multibody/inverse_kinematics/distance_constraint.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/multibody/inverse_kinematics/kinematic_constraint_utilities.h"
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
    const GripperBrickSystem<double>& gripper_brick_system, Finger finger,
    BrickFace brick_face, multibody::InverseKinematics* ik) {
  const multibody::Frame<double>& finger_link2 =
      gripper_brick_system.finger_link2_frame(finger);
  // position of Tip in the finger link 2 farme (F2).
  const Eigen::Vector3d p_F2Tip = gripper_brick_system.p_F2Tip();
  const multibody::Frame<double>& brick = gripper_brick_system.brick_frame();
  const Eigen::Vector3d box_size(0.025, 0.077, 0.077);
  Eigen::Vector3d p_BTip_lower = -box_size * 0.42;
  Eigen::Vector3d p_BTip_upper = box_size * 0.42;
  const double finger_tip_radius = gripper_brick_system.finger_tip_radius();
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
    const GripperBrickSystem<double>& gripper_brick_system,
    const systems::Context<double>& fixed_context, Finger finger,
    solvers::MathematicalProgram* prog,
    systems::Context<double>* plant_mutable_context,
    const VectorX<symbolic::Variable>& q) {
  const Eigen::Vector3d p_F2Tip = gripper_brick_system.p_F2Tip();
  const multibody::Frame<double>& finger_link2 =
      gripper_brick_system.finger_link2_frame(finger);
  const multibody::Frame<double>& brick = gripper_brick_system.brick_frame();
  Eigen::Vector3d p_BTip;
  gripper_brick_system.plant().CalcPointsPositions(fixed_context, finger_link2,
                                                   p_F2Tip, brick, &p_BTip);
  prog->AddConstraint(std::make_shared<multibody::PositionConstraint>(
                          &(gripper_brick_system.plant()), brick, p_BTip,
                          p_BTip, finger_link2, p_F2Tip, plant_mutable_context),
                      q);
}

void FixFingerPositionInBrickFrame(
    const GripperBrickSystem<double>& gripper_brick_system,
    const systems::Context<double>& fixed_context, Finger finger,
    multibody::InverseKinematics* ik) {
  FixFingerPositionInBrickFrame(gripper_brick_system, fixed_context, finger,
                                ik->get_mutable_prog(),
                                ik->get_mutable_context(), ik->q());
}

Eigen::MatrixXd InterpolateTrajectory(
    const GripperBrickSystem<double>& gripper_brick_system, int num_samples,
    const optional<Finger>& moving_finger_index,
    const Eigen::Ref<const Eigen::VectorXd>& q_start,
    const Eigen::Ref<const Eigen::VectorXd>& q_end,
    const systems::Context<double>& plant_fixed_context,
    const Eigen::Ref<const Eigen::VectorXd>& delta_q_max) {
  solvers::MathematicalProgram prog;
  const auto& diagram = gripper_brick_system.diagram();
  const auto& plant = gripper_brick_system.plant();
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
    for (Finger finger :
         {Finger::kFinger1, Finger::kFinger2, Finger::kFinger3}) {
      if (moving_finger_index.has_value() &&
          moving_finger_index.value() == finger) {
        SortedPair<geometry::GeometryId> geometry_pair(
            plant.GetCollisionGeometriesForBody(
                gripper_brick_system.brick_frame().body())[0],
            plant.GetCollisionGeometriesForBody(
                gripper_brick_system
                    .finger_link2_frame(moving_finger_index.value())
                    .body())[0]);
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
          FixFingerPositionInBrickFrame(gripper_brick_system,
                                        plant_fixed_context, finger, &prog,
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

void RotateBoxByCertainDegree(
    const GripperBrickSystem<double>& gripper_brick_system,
    double rotate_angle_lower, double rotate_angle_upper,
    const systems::Context<double>& plant_context,
    multibody::InverseKinematics* ik) {
  for (Finger finger : {Finger::kFinger1, Finger::kFinger2, Finger::kFinger3}) {
    FixFingerPositionInBrickFrame(gripper_brick_system, plant_context, finger,
                                  ik);
  }
  const Eigen::VectorXd q0 =
      gripper_brick_system.plant().GetPositions(plant_context);
  ik->get_mutable_prog()->AddBoundingBoxConstraint(
      q0(gripper_brick_system.brick_revolute_x_position_index()) +
          rotate_angle_lower,
      q0(gripper_brick_system.brick_revolute_x_position_index()) +
          rotate_angle_upper,
      ik->q()(gripper_brick_system.brick_revolute_x_position_index()));
}

/** Given the set of contacts between the fingers and the brick, impose the
 * static equilibrium as a nonlinear constraint, that the total force/torque
 * applied on the brick is 0.
 */
// class StaticEquilibriumNonlinearConstraint : public solvers::Constraint {
// public:
//  StaticEquilibriumNonlinearConstraint(
//      const GripperBrickSystem<double>& gripper_brick_system,
//      std::vector<std::pair<Finger, BrickFace>> finger_face_contacts,
//      systems::Context<double>* plant_mutable_context)
//      : solvers::Constraint(3,
//                            gripper_brick_system.plant().num_positions() +
//                                finger_face_contacts.size() * 2,
//                            Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()),
//        gripper_brick_system_{gripper_brick_system},
//        finger_face_contacts_(std::move(finger_face_contacts)),
//        plant_mutable_context_(plant_mutable_context) {
//    brick_mass_ = gripper_brick_system_.plant()
//                      .GetBodyByName("brick_link")
//                      .get_default_mass();
//  }
//
// private:
//  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
//              Eigen::VectorXd* y) const {
//    y->resize(3);
//    const auto& plant = gripper_brick_system_.plant();
//    multibody::internal::UpdateContextConfiguration(
//        plant_mutable_context_, plant, x.head(plant.num_positions()));
//    const math::RigidTransform<double> X_BW = plant.CalcRelativeTransform(
//        *plant_mutable_context_, gripper_brick_system.brick_frame(),
//        plant.world_frame());
//    const Eigen::Vector3d f_B =
//        X_BW.rotation() * Eigen::Vector3d(0, 0, -brick_mass_ * 9.81);
//    y->head<2>() = f_B.tail<2>();
//    (*y)(2) = 0;
//    for (int i = 0; i < static_cast<int>(finger_face_contacts_.size()); ++i) {
//      y->head<2>() += x.segment<2>(plant.num_positions() + i * 2);
//      Eigen::Vector3d p_BTip;
//      plant.CalcPointsPositions(*plant_mutable_context_,
//                                gripper_brick_system_.finger_link2_frame(
//                                    finger_face_contacts_[i].first),
//                                gripper_brick_system_.p_F2Tip(),
//                                gripper_brick_system_.brick_frame(), &p_BTip);
//    }
//  }
//
//  const GripperBrickSystem<double>& gripper_brick_system_;
//  double brick_mass_;
//  std::vector<std::pair<Finger, BrickFace>> finger_face_contacts_;
//  systems::Context<double>* plant_mutable_context_;
//};
//
// void AddStaticEquilibriumConstraint(
//    const GripperBrickSystem<double>& gripper_brick_system,
//    const std::vector<std::pair<int, BrickFace>>& finger_face_contacts,
//    double static_friction, systems::Context<double>* plant_mutable_context,
//    solvers::MathematicalProgram* prog) {
//  // force in the body frame.
//  auto f = prog->NewContinuousVariables<2, Eigen::Dynamic>(
//      2, finger_face_contacts.size());
//  // The total force on the brick should be 0.
//  // The total wrench on the brick should be 0.
//}

Eigen::VectorXd FindInitialPosture(
    const GripperBrickSystem<double>& gripper_brick_system,
    systems::Context<double>* plant_mutable_context,
    systems::Context<double>* diagram_context) {
  multibody::InverseKinematics ik0(gripper_brick_system.plant(),
                                   plant_mutable_context);

  // Finger 1 in +Z face. Finger 2 in -Z face. Finger 3 in +Y face.
  AddFingerTipInContactWithBrickFace(gripper_brick_system, Finger::kFinger1,
                                     BrickFace::kPosZ, &ik0);
  AddFingerTipInContactWithBrickFace(gripper_brick_system, Finger::kFinger2,
                                     BrickFace::kNegZ, &ik0);
  AddFingerTipInContactWithBrickFace(gripper_brick_system, Finger::kFinger3,
                                     BrickFace::kPosY, &ik0);
  // Add force equilibrium constraint.
  auto f1 = ik0.get_mutable_prog()->NewContinuousVariables<2>();
  auto f2 = ik0.get_mutable_prog()->NewContinuousVariables<2>();
  auto f3 = ik0.get_mutable_prog()->NewContinuousVariables<2>();

  // ik0.get_mutable_prog()->AddBoundingBoxConstraint(
  //    Eigen::Vector3d(-0.01, -0.01, -15.0 / 180 * M_PI),
  //    Eigen::Vector3d(0.01, 0.01, 15.0 / 180 * M_PI),
  //    Vector3<symbolic::Variable>(
  //        ik0.q()(gripper_brick_system.brick_translate_y_position_index()),
  //        ik0.q()(gripper_brick_system.brick_translate_z_position_index()),
  //        ik0.q()(gripper_brick_system.brick_revolute_x_position_index())));

  auto result = solvers::Solve(ik0.prog());
  std::cout << result.get_solution_result() << "\n";

  const Eigen::VectorXd q0 = result.GetSolution(ik0.q());
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q0);

  gripper_brick_system.diagram().Publish(*diagram_context);
  return q0;
}

Eigen::VectorXd FindPosture(
    const GripperBrickSystem<double>& gripper_brick_system,
    const std::vector<Finger>& fixed_fingers,
    const std::vector<std::pair<Finger, BrickFace>>& moving_fingers,
    systems::Context<double>* plant_mutable_context) {
  multibody::InverseKinematics ik(gripper_brick_system.plant(),
                                  plant_mutable_context);
  for (Finger fixed_finger : fixed_fingers) {
    FixFingerPositionInBrickFrame(gripper_brick_system, *plant_mutable_context,
                                  fixed_finger, &ik);
  }
  for (const auto& moving_finger : moving_fingers) {
    AddFingerTipInContactWithBrickFace(
        gripper_brick_system, moving_finger.first, moving_finger.second, &ik);
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
  const Eigen::VectorXd q1 = FindPosture(
      gripper_brick_system, {Finger::kFinger2, Finger::kFinger3},
      {{Finger::kFinger1, BrickFace::kNegY}}, plant_mutable_context);
  return q1;
}

Eigen::MatrixXd RotateBlockToPosture(
    const GripperBrickSystem<double>& gripper_brick_system,
    const Eigen::Ref<const Eigen::VectorXd>& q1, double angle_lower,
    double angle_upper, const Eigen::Ref<const Eigen::VectorXd>& delta_q_max,
    int num_samples, systems::Context<double>* plant_mutable_context) {
  multibody::InverseKinematics ik2(gripper_brick_system.plant(),
                                   plant_mutable_context);
  RotateBoxByCertainDegree(gripper_brick_system, angle_lower, angle_upper,
                           *plant_mutable_context, &ik2);
  const solvers::MathematicalProgramResult result =
      solvers::Solve(ik2.prog(), q1);
  std::cout << result.get_solution_result() << "\n";
  const Eigen::VectorXd q2 = result.GetSolution(ik2.q());

  gripper_brick_system.plant().SetPositions(plant_mutable_context, q1);

  // Now find the samples from q1 to q2
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q1);
  const Eigen::MatrixXd q_move =
      InterpolateTrajectory(gripper_brick_system, num_samples, {}, q1, q2,
                            *plant_mutable_context, delta_q_max);
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
  const Eigen::VectorXd q2 = FindPosture(
      gripper_brick_system, {Finger::kFinger1, Finger::kFinger2},
      {{Finger::kFinger3, BrickFace::kNegZ}}, plant_mutable_context);
  return q2;
}

int DoMain() {
  GripperBrickSystem<double> gripper_brick_system;
  auto print_joint_start_index = [&gripper_brick_system](
                                     const std::string& joint_name) {
    const int position_start_index = gripper_brick_system.plant()
                                         .GetJointByName(joint_name)
                                         .position_start();
    std::cout << joint_name << " starts at " << position_start_index << "\n";
  };
  print_joint_start_index("finger1_ShoulderJoint");
  print_joint_start_index("finger1_ElbowJoint");
  print_joint_start_index("finger2_ShoulderJoint");
  print_joint_start_index("finger2_ElbowJoint");
  print_joint_start_index("finger3_ShoulderJoint");
  print_joint_start_index("finger3_ElbowJoint");
  print_joint_start_index("brick_translate_y_joint");
  print_joint_start_index("brick_translate_z_joint");
  print_joint_start_index("brick_revolute_x_joint");

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
      0.3 * Eigen::VectorXd::Ones(gripper_brick_system.plant().num_positions());
  const Eigen::MatrixXd q_move1 =
      InterpolateTrajectory(gripper_brick_system, 9, Finger::kFinger1, q0, q1,
                            *plant_mutable_context, delta_q_max);
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
      gripper_brick_system, 8, Finger::kFinger3,
      q_move2.col(q_move2.cols() - 1), q3, *plant_mutable_context, delta_q_max);
  for (int i = 0; i < q_move3.cols(); ++i) {
    VisualizePosture(gripper_brick_system, q_move3.col(i),
                     plant_mutable_context, diagram_context.get());
    sleep(1);
  }

  // Move finger 2 to negY face.
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q3);
  const Eigen::VectorXd q4 = FindPosture(
      gripper_brick_system, {Finger::kFinger1, Finger::kFinger3},
      {{Finger::kFinger2, BrickFace::kNegY}}, plant_mutable_context);
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q4);
  delta_q_max =
      0.2 * Eigen::VectorXd::Ones(gripper_brick_system.plant().num_positions());
  const Eigen::MatrixXd q_move4 =
      InterpolateTrajectory(gripper_brick_system, 9, Finger::kFinger2, q3, q4,
                            *plant_mutable_context, delta_q_max);
  for (int i = 0; i < q_move4.cols(); ++i) {
    VisualizePosture(gripper_brick_system, q_move4.col(i),
                     plant_mutable_context, diagram_context.get());
    sleep(1);
  }

  // Move finger 1 to posZ face
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q4);
  const Eigen::VectorXd q5 = FindPosture(
      gripper_brick_system, {Finger::kFinger2, Finger::kFinger3},
      {{Finger::kFinger1, BrickFace::kPosZ}}, plant_mutable_context);
  gripper_brick_system.plant().SetPositions(plant_mutable_context, q5);
  delta_q_max = 0.25 * Eigen::VectorXd::Ones(
                           gripper_brick_system.plant().num_positions());
  const Eigen::MatrixXd q_move5 =
      InterpolateTrajectory(gripper_brick_system, 8, Finger::kFinger1, q4, q5,
                            *plant_mutable_context, delta_q_max);
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

  std::stringstream ss;
  ss << std::setprecision(20);
  std::cout << ss.str() << "\n";

  return 0;
}
}  // namespace examples
}  // namespace drake

int main() { drake::examples::DoMain(); }
