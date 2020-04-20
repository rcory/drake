#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/examples/planar_gripper/contact_force_qp.h"
#include "drake/examples/planar_gripper/planar_gripper.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/lcm/drake_lcm.h"
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
DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_double(simulation_time, 5.0,
              "Desired duration of the simulation in seconds.");
DEFINE_double(time_step, 1e-3,
              "If greater than zero, the plant is modeled as a system with "
              "discrete updates and period equal to this time_step. "
              "If 0, the plant is modeled as a continuous system.");
DEFINE_string(brick_type, "planar", "The brick type: {pin, planar}.");
DEFINE_double(brick_y0, 0.05, "The brick's initial y position (m).");
DEFINE_double(brick_z0, -0.05, "The brick's initial z position (m).");
DEFINE_double(brick_theta0, -M_PI_4 + 0.2,
              "The brick's initial revolute-x rotation (rad).");
DEFINE_double(brick_thetaf, M_PI_4, "The final rotation of the brick. (rad)");
DEFINE_string(control_task_type, "track",
              "Control task type: {track, regulate}.");
DEFINE_bool(test, false,
            "If true, checks the simulation result against a known value.");

int DoMain() {
  systems::DiagramBuilder<double> builder;

  auto planar_gripper =
      builder.AddSystem<PlanarGripper>(FLAGS_time_step, ControlType::kTorque);
  planar_gripper->set_brick_floor_penetration(0);

  BrickType brick_type;
  VectorX<double> brick_initial_positions;
  if (FLAGS_brick_type == "planar") {
    planar_gripper->SetupPlanarBrick("horizontal");
    brick_type = BrickType::PlanarBrick;
    std::map<std::string, double> position_map;
    position_map["brick_translate_x_joint"] = 0;
    position_map["brick_translate_y_joint"] = FLAGS_brick_y0;
    position_map["brick_translate_z_joint"] = FLAGS_brick_z0;
    position_map["brick_revolute_x_joint"] = FLAGS_brick_theta0;
    brick_initial_positions =
        planar_gripper->MakeBrickPositionVector(position_map);
  } else if (FLAGS_brick_type == "pin") {
    std::string orientation = "vertical";  // {vertical, horizontal}
    planar_gripper->SetupPinBrick(orientation);
    brick_type = BrickType::PinBrick;
    std::map<std::string, double> position_map;
    if (orientation == "horizontal") {
      position_map["brick_translate_x_joint"] = 0;
    }
    position_map["brick_revolute_x_joint"] = FLAGS_brick_theta0;
    brick_initial_positions =
        planar_gripper->MakeBrickPositionVector(position_map);
  } else {
    throw std::logic_error("Unknown brick type.");
  }
  planar_gripper->zero_gravity();
  planar_gripper->Finalize();

  // Setup the QP controller parameters.
  const Eigen::Matrix2d Kp_t = Eigen::Vector2d(150, 150).asDiagonal();
  const Eigen::Matrix2d Kd_t = Eigen::Vector2d(50, 50).asDiagonal();
  double Kp_r = 150;
  double Kd_r = 50;
  double weight_a_error = 1;
  double weight_thetaddot_error = 1;
  double weight_f_Cb_B = 1;
  double mu = 1.0;
  double brick_translational_damping = 0;

  // Get the brick's Ixx moment of inertia (i.e., around the pinned axis).
  const int kIxx_index = 0;
  double I_B = planar_gripper->GetBrickMoments()(kIxx_index);
  double mass_B = planar_gripper->GetBrickMass();
  double brick_revolute_damping = planar_gripper->GetBrickPinJointDamping();

  auto qp_controller = builder.AddSystem<InstantaneousContactForceQPController>(
      brick_type, &planar_gripper->get_multibody_plant(), Kp_t, Kd_t, Kp_r,
      Kd_r, weight_a_error, weight_thetaddot_error, weight_f_Cb_B, mu,
      brick_translational_damping, brick_revolute_damping, I_B, mass_B);

  // Connect the QP controller.
  builder.Connect(planar_gripper->GetOutputPort("plant_state"),
                  qp_controller->get_input_port_estimated_state());
  builder.Connect(qp_controller->get_output_port_brick_control(),
                  planar_gripper->GetInputPort("spatial_force"));

  // To visualize the spatial forces on the brick.
  auto viz_converter = builder.AddSystem<ExternalSpatialToSpatialViz>(
      planar_gripper->get_multibody_plant(), planar_gripper->get_brick_index(),
      1.0);
  builder.Connect(qp_controller->get_output_port_brick_control(),
                  viz_converter->get_input_port(0));
  builder.Connect(planar_gripper->GetOutputPort("brick_state"),
                  viz_converter->get_input_port(1));
  lcm::DrakeLcm drake_lcm;
  multibody::ConnectSpatialForcesToDrakeVisualizer(
      &builder, planar_gripper->get_multibody_plant(),
      viz_converter->get_output_port(0), &drake_lcm);

  // Specify the finger/contact face pairing.
  std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>
      finger_face_assignments;

  finger_face_assignments.emplace(
      Finger::kFinger1,
      std::make_pair(BrickFace::kNegY, Eigen::Vector2d(-0.05, 0)));
  finger_face_assignments.emplace(
      Finger::kFinger2,
      std::make_pair(BrickFace::kPosY, Eigen::Vector2d(0.05, 0)));
  finger_face_assignments.emplace(
      Finger::kFinger3,
      std::make_pair(BrickFace::kNegZ, Eigen::Vector2d(0, -0.05)));

  auto finger_face_assignments_source = builder.AddSystem<
      systems::ConstantValueSource<double>>(
      Value<std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>>(
          finger_face_assignments));
  builder.Connect(finger_face_assignments_source->get_output_port(0),
                  qp_controller->get_input_port_finger_face_assignments());

  VectorX<double> brick_planned_acceleration_vector;
  trajectories::PiecewisePolynomial<double> brick_planned_state_traj;
  VectorX<double> des_state_vec;
  if (brick_type == BrickType::PlanarBrick) {
    // thetaddot_planned is 0. Use a constant source.
    brick_planned_acceleration_vector = Eigen::Vector3d::Zero();

    // The planned theta trajectory is from 0 to thetaf degrees in 1 second.
    // This is used if the control task type is `track`.
    brick_planned_state_traj =
        trajectories::PiecewisePolynomial<double>::FirstOrderHold(
            {0, 1}, {Eigen::Vector3d(0, 0, 0),
                     Eigen::Vector3d(0, 0, FLAGS_brick_thetaf)});

    // Defines a constant state target...{y, z, theta, ydot, zdot, thetadot}.
    // This is used if the control task type is `regulate`.
    des_state_vec = VectorX<double>::Zero(6);
    des_state_vec << 0, 0, FLAGS_brick_thetaf, 0, 0, 0;
  } else {
    // thetaddot_planned is 0. Use a constant source.
    brick_planned_acceleration_vector = Vector1d(0);

    // The planned theta trajectory is from 0 to thetaf degrees in 1 second.
    // This is used if the control task type is `track`.
    brick_planned_state_traj =
        trajectories::PiecewisePolynomial<double>::FirstOrderHold(
            {0, 1}, {Vector1d(0), Vector1d(FLAGS_brick_thetaf)});

    // Defines a constant state target...{theta, thetadot}. This is used if
    // the control task type is `regulate`.
    des_state_vec = Eigen::Vector2d(FLAGS_brick_thetaf, 0);
  }
  auto brick_acceleration_planned_source =
      builder.AddSystem<systems::ConstantVectorSource<double>>(
          brick_planned_acceleration_vector);
  builder.Connect(brick_acceleration_planned_source->get_output_port(),
                  qp_controller->get_input_port_desired_brick_acceleration());

  if (FLAGS_control_task_type == "track") {
    auto brick_state_traj_source =
        builder.AddSystem<systems::TrajectorySource<double>>(
            brick_planned_state_traj, 1 /* take 1st derivatives */);
    builder.Connect(brick_state_traj_source->get_output_port(),
                    qp_controller->get_input_port_desired_state());
  } else {  // regulate
    auto brick_state_traj_source =
        builder.AddSystem<systems::ConstantVectorSource<double>>(des_state_vec);
    builder.Connect(brick_state_traj_source->get_output_port(),
                    qp_controller->get_input_port_desired_state());
  }

  /** Apply zero torque to the joint motor. */
  auto gripper_actuation_source =
      builder.AddSystem<systems::TrajectorySource<double>>(
          trajectories::PiecewisePolynomial<double>::ZeroOrderHold(
              {0, 1}, {Eigen::Matrix<double, 6, 1>::Zero(),
                       Eigen::Matrix<double, 6, 1>::Zero()}));
  builder.Connect(gripper_actuation_source->get_output_port(),
                  planar_gripper->GetInputPort("torque_control_u"));

  // Publish body frames.
  auto frame_viz = builder.AddSystem<FrameViz>(
      planar_gripper->get_multibody_plant(), &drake_lcm, 1.0 / 60.0, false);
  builder.Connect(planar_gripper->GetOutputPort("plant_state"),
                  frame_viz->get_input_port(0));

  math::RigidTransformd goal_frame;
  goal_frame.set_rotation(math::RollPitchYaw<double>(FLAGS_brick_thetaf, 0, 0));
  PublishFramesToLcm("GOAL_FRAME", {goal_frame}, {"goal"}, &drake_lcm);

  // Connect drake visualizer.
  geometry::ConnectDrakeVisualizer(
      &builder, planar_gripper->get_mutable_scene_graph(),
      planar_gripper->GetOutputPort("pose_bundle"));

  auto diagram = builder.Build();

  // Set the initial conditions for the planar-gripper.
  std::map<std::string, double> init_gripper_pos_map;
  init_gripper_pos_map["finger1_BaseJoint"] = -1;
  init_gripper_pos_map["finger1_MidJoint"] = -1;
  init_gripper_pos_map["finger2_BaseJoint"] = -1;
  init_gripper_pos_map["finger2_MidJoint"] = -1;
  init_gripper_pos_map["finger3_BaseJoint"] = -1;
  init_gripper_pos_map["finger3_MidJoint"] = -1;

  auto gripper_initial_positions =
      planar_gripper->MakeGripperPositionVector(init_gripper_pos_map);

  // Create a context for the diagram.
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& planar_gripper_context =
      diagram->GetMutableSubsystemContext(*planar_gripper,
                                          diagram_context.get());

  planar_gripper->SetGripperPosition(&planar_gripper_context,
                                     gripper_initial_positions);
  planar_gripper->SetBrickPosition(&planar_gripper_context,
                                   brick_initial_positions);

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);

  // TODO(rcory) Implement a proper unit test once all shared parameters are
  //  moved to a YAML file.
  if (FLAGS_test) {
    VectorX<double> x_known(20);
    x_known << 0, -1, -1, -1, 1.69e-05, -1, -1, -1, -3.19e-05,
        78526.99e-5, 0, 0, 0, 0, -5.92e-05, 0, 0, 0, 10.23e-5,
        41.43e-5;
    const auto& post_sim_context = simulator.get_context();
    const auto& post_plant_context = diagram->GetSubsystemContext(
        planar_gripper->get_mutable_multibody_plant(), post_sim_context);
    const auto post_plant_state =
        planar_gripper->get_multibody_plant().GetPositionsAndVelocities(
            post_plant_context);
    // Check to within an arbitrary threshold.
    DRAKE_DEMAND(x_known.isApprox(post_plant_state, 1e-6));
  }

  return 0;
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::planar_gripper::DoMain();
}
