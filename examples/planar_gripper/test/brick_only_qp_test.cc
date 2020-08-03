#include <memory>

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/examples/planar_gripper/contact_force_qp.h"
#include "drake/examples/planar_gripper/planar_gripper.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/plant/spatial_forces_to_lcm.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_value_source.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/trajectory_source.h"

namespace drake {
namespace examples {
namespace planar_gripper {

using Eigen::Vector2d;
using Eigen::VectorXd;
using geometry::SceneGraph;

// The nominal planar dimension (length & width) of the brick.
constexpr double kBoxDimension = 0.1;

// This file tests the QP contact force planner over a variety of planar brick
// simulations, where the planned force is directly applied as a spatial force
// on the brick body at a set of specified contact locations. Note that in
// these tests no planar gripper fingers are used.

class QPControlTest : public testing::Test {
 protected:
  void BuildDiagram(
      BrickType brick_type, const VectorXd& brick_q0, const VectorXd& brick_qf,
      const std::unordered_map<Finger, BrickFaceInfo>& finger_face_assignments,
      const std::string& control_task_type, bool add_floor) {
    systems::DiagramBuilder<double> builder;

    planar_gripper_ =
        builder.AddSystem<PlanarGripper>(1e-3, ControlType::kTorque, add_floor);
    planar_gripper_->set_floor_coef_static_friction(0.5);
    planar_gripper_->set_floor_coef_kinetic_friction(0.5);
    planar_gripper_->set_brick_floor_penetration(1e-7);

    VectorX<double> brick_initial_positions;
    if (brick_type == BrickType::PlanarBrick) {
      planar_gripper_->SetupPlanarBrick("vertical");
      std::map<std::string, double> position_map;
      position_map["brick_translate_y_joint"] = brick_q0(0);
      position_map["brick_translate_z_joint"] = brick_q0(1);
      position_map["brick_revolute_x_joint"] = brick_q0(2);
      brick_initial_positions =
          planar_gripper_->MakeBrickPositionVector(position_map);
    } else if (brick_type == BrickType::PinBrick) {
      std::string orientation = "vertical";  // {vertical, horizontal}
      planar_gripper_->SetupPinBrick(orientation);
      brick_type = BrickType::PinBrick;
      std::map<std::string, double> position_map;
      position_map["brick_revolute_x_joint"] = brick_q0(0);
      brick_initial_positions =
          planar_gripper_->MakeBrickPositionVector(position_map);
    } else {
      throw std::logic_error("Unknown brick type.");
    }
    planar_gripper_->set_penetration_allowance(1e-3);
    planar_gripper_->set_stiction_tolerance(1e-3);
    planar_gripper_->zero_gravity();
    planar_gripper_->Finalize();

    // Setup the QP controller parameters.
    // Translation gains (y, z).
    const Vector2d Kp_t = Vector2d(150, 150);
    const Vector2d Kd_t = Vector2d(50, 50);
    const Vector2d Ki_t = add_floor ? Vector2d(500, 500) : Vector2d::Zero();
    const Vector2d Ki_t_sat = Vector2d(5e-4, 5e-4);
    const double ki_r = add_floor ? 10e3 : 0;
    const double ki_r_sat = 4e-3;
    const double kp_r = 3e3;
    const double kd_r = 1e3;
    const double weight_a_error = 1;
    const double weight_thetaddot_error = 1;
    const double weight_f_Cb_B = 1e3;
    const double mu = 1.0;
    const double brick_translational_damping = 0;

    // Get the brick's Ixx moment of inertia (i.e., around the pinned axis).
    const int kIxx_index = 0;
    double I_B = planar_gripper_->GetBrickMoments()(kIxx_index);
    double mass_B = planar_gripper_->GetBrickMass();
    double brick_revolute_damping = planar_gripper_->GetBrickPinJointDamping();

    auto qp_controller =
        builder.AddSystem<InstantaneousContactForceQPController>(
            brick_type, &planar_gripper_->get_multibody_plant(), Kp_t, Kd_t,
            Ki_t, Ki_t_sat, kp_r, kd_r, ki_r, ki_r_sat, weight_a_error,
            weight_thetaddot_error, weight_f_Cb_B, mu,
            brick_translational_damping, brick_revolute_damping, I_B, mass_B);

    // Connect the QP controller.
    builder.Connect(planar_gripper_->GetOutputPort("plant_state"),
                    qp_controller->get_input_port_estimated_state());
    builder.Connect(qp_controller->get_output_port_brick_control(),
                    planar_gripper_->GetInputPort("spatial_force"));

    // To visualize the spatial forces on the brick.
    auto viz_converter = builder.AddSystem<ExternalSpatialToSpatialViz>(
        planar_gripper_->get_multibody_plant(),
        planar_gripper_->get_brick_index(), 1.0);
    builder.Connect(qp_controller->get_output_port_brick_control(),
                    viz_converter->get_input_port(0));
    builder.Connect(planar_gripper_->GetOutputPort("brick_state"),
                    viz_converter->get_input_port(1));
    multibody::ConnectSpatialForcesToDrakeVisualizer(
        &builder, planar_gripper_->get_multibody_plant(),
        viz_converter->get_output_port(0), &drake_lcm_);

    auto finger_face_assignments_source =
        builder.AddSystem<systems::ConstantValueSource<double>>(
            Value<std::unordered_map<Finger, BrickFaceInfo>>(
                finger_face_assignments));

    builder.Connect(finger_face_assignments_source->get_output_port(0),
                    qp_controller->get_input_port_finger_face_assignments());

    // The planned trajectory time span is 2 seconds. This is used if the
    // control task type is set to `track`.
    const double tspan = 1.0;  // time span
    DRAKE_DEMAND(brick_q0.size() == brick_qf.size());

    VectorX<double> brick_planned_acceleration_vector;
    VectorX<double> des_state_vec;
    math::RigidTransformd goal_frame;
    std::vector<double> T = {-0.5, 0, tspan, tspan + 0.5};  // break points
    std::vector<MatrixX<double>> Y;  // polynomial samples

    if (brick_type == BrickType::PlanarBrick) {
      Y = std::vector<MatrixX<double>>(T.size(), MatrixX<double>::Zero(3, 1));
      Y[0](0, 0) = brick_q0(0);
      Y[0](1, 0) = brick_q0(1);
      Y[0](2, 0) = brick_q0(2);

      Y[1](0, 0) = brick_q0(0);
      Y[1](1, 0) = brick_q0(1);
      Y[1](2, 0) = brick_q0(2);

      Y[2](0, 0) = brick_qf(0);
      Y[2](1, 0) = brick_qf(1);
      Y[2](2, 0) = brick_qf(2);

      Y[3](0, 0) = brick_qf(0);
      Y[3](1, 0) = brick_qf(1);
      Y[3](2, 0) = brick_qf(2);

      // thetaddot_planned is 0. Use a constant source.
      brick_planned_acceleration_vector = Eigen::Vector3d::Zero();
      goal_frame.set_translation(Vector3d(0, brick_qf(0), brick_qf(1)));
      goal_frame.set_rotation(math::RollPitchYaw<double>(brick_qf(2), 0, 0));

      // Defines a constant state target...{y, z, theta, ydot, zdot, thetadot}.
      // This is used if the control task type is `regulate`.
      des_state_vec = VectorX<double>::Zero(6);
      des_state_vec.head<3>() = brick_qf;
    } else {  // pinned brick
      Y = std::vector<MatrixX<double>>(T.size(), MatrixX<double>::Zero(1, 1));
      Y[0](0, 0) = brick_q0(0);
      Y[1](0, 0) = brick_q0(0);
      Y[2](0, 0) = brick_qf(0);
      Y[3](0, 0) = brick_qf(0);

      // thetaddot_planned is 0. Use a constant source.
      brick_planned_acceleration_vector = Vector1d(0);
      goal_frame.set_rotation(math::RollPitchYaw<double>(brick_qf(0), 0, 0));

      // Defines a constant state target...{theta, thetadot}. This is used if
      // the control task type is `regulate`.
      des_state_vec = Vector2d(brick_qf(0), 0);
    }
    trajectories::PiecewisePolynomial<double> brick_planned_state_traj =
        trajectories::PiecewisePolynomial<double>::CubicShapePreserving(
            T, Y, true /* zero end derivatives */);
    auto brick_acceleration_planned_source =
        builder.AddSystem<systems::ConstantVectorSource<double>>(
            brick_planned_acceleration_vector);
    builder.Connect(brick_acceleration_planned_source->get_output_port(),
                    qp_controller->get_input_port_desired_brick_acceleration());

    if (control_task_type == "track") {
      auto brick_state_traj_source =
          builder.AddSystem<systems::TrajectorySource<double>>(
              brick_planned_state_traj, 1 /* take 1st derivatives */,
              true /* zero derivatives beyond limits */);
      builder.Connect(brick_state_traj_source->get_output_port(),
                      qp_controller->get_input_port_desired_state());
    } else {  // regulate
      auto brick_state_traj_source =
          builder.AddSystem<systems::ConstantVectorSource<double>>(
              des_state_vec);
      builder.Connect(brick_state_traj_source->get_output_port(),
                      qp_controller->get_input_port_desired_state());
    }

    // Apply zero torque to the joint motors.
    auto gripper_actuation_source =
        builder.AddSystem<systems::TrajectorySource<double>>(
            trajectories::PiecewisePolynomial<double>::ZeroOrderHold(
                {0, 1}, {Eigen::Matrix<double, 6, 1>::Zero(),
                         Eigen::Matrix<double, 6, 1>::Zero()}));
    builder.Connect(gripper_actuation_source->get_output_port(),
                    planar_gripper_->GetInputPort("torque_control_u"));

    // Publish body frames.
    auto frame_viz = builder.AddSystem<FrameViz>(
        planar_gripper_->get_multibody_plant(), &drake_lcm_, 1.0 / 60.0, false);
    builder.Connect(planar_gripper_->GetOutputPort("plant_state"),
                    frame_viz->get_input_port(0));
    PublishFramesToLcm("GOAL_FRAME", {goal_frame}, {"goal"}, &drake_lcm_);

    // Connect drake visualizer.
    geometry::ConnectDrakeVisualizer(
        &builder, planar_gripper_->get_mutable_scene_graph(),
        planar_gripper_->GetOutputPort("pose_bundle"));

    // Connect contact results.
    ConnectContactResultsToDrakeVisualizer(
        &builder, planar_gripper_->get_mutable_multibody_plant(),
        planar_gripper_->GetOutputPort("contact_results"));

    diagram_ = builder.Build();

    // Set the initial conditions for the planar-gripper. Pose the fingers
    // such that they are out of the way of the brick's workspace.
    std::map<std::string, double> init_gripper_pos_map;
    init_gripper_pos_map["finger1_BaseJoint"] = -1.0;
    init_gripper_pos_map["finger1_MidJoint"] = -1.0;
    init_gripper_pos_map["finger2_BaseJoint"] = -1.0;
    init_gripper_pos_map["finger2_MidJoint"] = -1.0;
    init_gripper_pos_map["finger3_BaseJoint"] = -1.0;
    init_gripper_pos_map["finger3_MidJoint"] = -1.0;

    auto gripper_initial_positions =
        planar_gripper_->MakeGripperPositionVector(init_gripper_pos_map);

    // Create a context for the diagram.
    diagram_context_ = diagram_->CreateDefaultContext();
    systems::Context<double>& planar_gripper_context =
        diagram_->GetMutableSubsystemContext(*planar_gripper_,
                                             diagram_context_.get());

    planar_gripper_->SetGripperPosition(&planar_gripper_context,
                                        gripper_initial_positions);
    planar_gripper_->SetBrickPosition(&planar_gripper_context,
                                      brick_initial_positions);
  }

  VectorXd Simulate() {
    // Simulate the system.
    systems::Simulator<double> simulator(*diagram_,
                                         std::move(diagram_context_));
    simulator.set_target_realtime_rate(0);
    simulator.Initialize();
    simulator.AdvanceTo(4);  // Run to approximate convergence.

    // Check the steady state actual force against the desired force.
    const auto& post_sim_context = simulator.get_context();
    const auto& post_plant_context = diagram_->GetSubsystemContext(
        planar_gripper_->get_multibody_plant(), post_sim_context);
    const auto post_brick_state =
        planar_gripper_->get_multibody_plant().GetPositionsAndVelocities(
            post_plant_context, planar_gripper_->get_brick_index());
    return post_brick_state;
  }

  PlanarGripper* planar_gripper_{nullptr};
  std::unique_ptr<systems::Diagram<double>> diagram_;
  std::unique_ptr<systems::Context<double>> diagram_context_;
  lcm::DrakeLcm drake_lcm_;
};

std::unordered_map<Finger, BrickFaceInfo> MakeFingerFaceAssignments(
    const std::vector<Finger>& fingers) {
  std::unordered_map<Finger, BrickFaceInfo> finger_face_assignments;
  for (auto& finger : fingers) {
    if (finger == Finger::kFinger1) {
      finger_face_assignments.emplace(
          Finger::kFinger1,
          BrickFaceInfo(BrickFace::kNegY, Vector2d(-kBoxDimension / 2, 0),
                        true));
    } else if (finger == Finger::kFinger2) {
      finger_face_assignments.emplace(
          Finger::kFinger2,
          BrickFaceInfo(BrickFace::kPosY, Vector2d(kBoxDimension / 2, 0),
                        true));
    } else if (finger == Finger::kFinger3) {
      finger_face_assignments.emplace(
          Finger::kFinger3,
          BrickFaceInfo(BrickFace::kNegZ, Vector2d(0, -kBoxDimension / 2),
                        true));
    }
  }
  return finger_face_assignments;
}

// The following tests attempt to track a straight path from the brick's initial
// state `x0` to a goal final state `xf`. These tests differ in the number of
// contacts used, the type of brick that is simulated (planar, pinned), and
// whether a floor is present or not.

TEST_F(QPControlTest, PlanarBrickWithFloorOneFingerTranslationTest) {
  VectorXd brick_x0 = VectorXd::Zero(6);
  brick_x0.head<3>() = Vector3d(0, 0.025, 0);
  VectorXd brick_xf = VectorXd::Zero(6);
  auto finger_face_assignments = MakeFingerFaceAssignments({Finger::kFinger3});
  BuildDiagram(BrickType::PlanarBrick, brick_x0.head<3>(),
               brick_xf.head<3>(), finger_face_assignments, "track",
               true /* adds floor */);
  // The QP controller should return a zero value force for the single finger
  // case, and therefore the brick should remain at x0.
  EXPECT_TRUE(CompareMatrices(brick_x0, Simulate(), 1e-12));
}

TEST_F(QPControlTest, PlanarBrickWithFloorTwoFingerTranslationTest) {
  VectorXd brick_xf = VectorXd::Zero(6);
  auto finger_face_assignments =
      MakeFingerFaceAssignments({Finger::kFinger1, Finger::kFinger3});
  finger_face_assignments.at(Finger::kFinger1).brick_face = BrickFace::kPosZ;
  finger_face_assignments.at(Finger::kFinger1).p_BCb =
      Vector2d(0, kBoxDimension / 2);
  BuildDiagram(BrickType::PlanarBrick, Vector3d(0.025, 0.025, 0) /* brick q0 */,
               brick_xf.head<3>(), finger_face_assignments, "track",
               true /* adds floor */);
  EXPECT_TRUE(CompareMatrices(brick_xf, Simulate(), 2.5e-3));
}

TEST_F(QPControlTest, PlanarBrickWithFloorTwoFingerRotationTest) {
  VectorXd brick_xf(6);
  brick_xf << 0, 0, M_PI_4, 0, 0, 0;
  auto finger_face_assignments = MakeFingerFaceAssignments(
      {Finger::kFinger2, Finger::kFinger3});
  finger_face_assignments.at(Finger::kFinger2).brick_face = BrickFace::kPosZ;
  finger_face_assignments.at(Finger::kFinger2).p_BCb =
      Vector2d(0, kBoxDimension / 2);
  BuildDiagram(BrickType::PlanarBrick,
               Vector3d(0, 0, -M_PI_4 + 0.2) /* brick q0 */,
               brick_xf.head<3>(), finger_face_assignments, "track",
               true /* adds floor */);
  EXPECT_TRUE(CompareMatrices(brick_xf, Simulate(), 2.5e-3));
}

TEST_F(QPControlTest, PlanarBrickNoFloorThreeFingerTest) {
  VectorXd brick_xf(6);
  brick_xf << 0, 0, M_PI_4, 0, 0, 0;
  auto finger_face_assignments = MakeFingerFaceAssignments(
      {Finger::kFinger1, Finger::kFinger2, Finger::kFinger3});
  BuildDiagram(BrickType::PlanarBrick,
               Vector3d(0.05, -0.05, -M_PI_4 + 0.2) /* brick q0 */,
               brick_xf.head<3>(), finger_face_assignments, "track",
               false /* no floor */);
  EXPECT_TRUE(CompareMatrices(brick_xf, Simulate(), 1e-4));
}

TEST_F(QPControlTest, PinnedBrickNoFloorThreeFingerTest) {
  Vector2d brick_xf(M_PI_4, 0);
  auto finger_face_assignments = MakeFingerFaceAssignments(
      {Finger::kFinger1, Finger::kFinger2, Finger::kFinger3});
  BuildDiagram(BrickType::PinBrick, Vector1d(-M_PI_4 + 0.2) /* brick q0 */,
               brick_xf.head<1>(), finger_face_assignments, "track",
               false /* no floor */);
  EXPECT_TRUE(CompareMatrices(brick_xf, Simulate(), 1e-4));
}

TEST_F(QPControlTest, PinnedBrickNoFloorThreeFingerTest2) {
  Vector2d brick_xf(M_PI_4, 0);
  auto finger_face_assignments = MakeFingerFaceAssignments(
      {Finger::kFinger1, Finger::kFinger2, Finger::kFinger3});
  BuildDiagram(BrickType::PinBrick, Vector1d(-M_PI_4 + 0.2) /* brick q0 */,
               brick_xf.head<1>(), finger_face_assignments, "regulate",
               false /* no floor */);
  EXPECT_TRUE(CompareMatrices(brick_xf, Simulate(), 1e-4));
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
