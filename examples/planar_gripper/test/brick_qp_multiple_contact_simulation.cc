// TODO(rcory) This file will be deprecated soon.
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/common/proto/call_python.h"
#include "drake/examples/planar_gripper/contact_force_qp.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_value_source.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/signal_logger.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/examples/planar_gripper/planar_gripper.h"
#include "drake/multibody/plant/spatial_forces_to_lcm.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/multibody/tree/revolute_joint.h"

namespace drake {
namespace examples {
namespace planar_gripper {
DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_double(simulation_time, 2.0,
              "Desired duration of the simulation in seconds.");
DEFINE_double(time_step, 1e-4,
              "If greater than zero, the plant is modeled as a system with "
              "discrete updates and period equal to this time_step. "
              "If 0, the plant is modeled as a continuous system.");
DEFINE_string(brick_type, "pin", "The brick type: {pin, planar}.");

int DoMain() {
  systems::DiagramBuilder<double> builder;

  auto planar_gripper =
      builder.AddSystem<PlanarGripper>(FLAGS_time_step, false);
  planar_gripper->zero_gravity(true);
  planar_gripper->set_brick_floor_penetration(0);
  BrickType brick_type;
  if (FLAGS_brick_type == "planar") {
    planar_gripper->SetupPlanarBrick("horizontal");
    brick_type = BrickType::PlanarBrick;
  } else if (FLAGS_brick_type == "pin") {
    planar_gripper->SetupPinBrick("horizontal");
    brick_type = BrickType::PinBrick;
  } else {
    throw std::logic_error("Unknown brick type.");
  }
  planar_gripper->Finalize();

  // QP controller
  const Eigen::Matrix2d Kp1 = Eigen::Vector2d(10, 10).asDiagonal();
  const Eigen::Matrix2d Kd1 = Eigen::Vector2d(10, 10).asDiagonal();
  double Kp2 = 10;
  double Kd2 = 10;
  double weight_a = 10;
  double weight_thetaddot_error = 1;
  double weight_f_Cb_B = 0.01;
  double I_B = dynamic_cast<const multibody::RigidBody<double>&>(
      planar_gripper->get_multibody_plant().GetFrameByName("brick_link").body())
      .default_rotational_inertia()
      .get_moments()(0);
  double mass_B = dynamic_cast<const multibody::RigidBody<double>&>(
      planar_gripper->get_multibody_plant().GetFrameByName("brick_link").body())
      .default_mass();
  double brick_revolute_damping =
      planar_gripper->get_multibody_plant()
          .GetJointByName<multibody::RevoluteJoint>("brick_revolute_x_joint")
          .damping();

  auto qp_controller = builder.AddSystem<InstantaneousContactForceQPController>(
      brick_type, &planar_gripper->get_multibody_plant(),
      Kp1, Kd1, Kp2, Kd2, weight_a, weight_thetaddot_error,
      weight_f_Cb_B, 1.0, 0, brick_revolute_damping, I_B, mass_B);

  // Connect the QP controller
  builder.Connect(planar_gripper->GetOutputPort("plant_state"),
                  qp_controller->get_input_port_estimated_state());
  builder.Connect(qp_controller->get_output_port_brick_control(),
                  planar_gripper->GetInputPort("spatial_force"));

  // Specify the finger/contact face pairing.
  std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>
      finger_face_assignments;

  finger_face_assignments.emplace(
      Finger::kFinger1,
      std::make_pair(BrickFace::kNegZ, Eigen::Vector2d(0, -0.023)));
  finger_face_assignments.emplace(
      Finger::kFinger2,
      std::make_pair(BrickFace::kPosY, Eigen::Vector2d(0.023, 0)));
  finger_face_assignments.emplace(
      Finger::kFinger3,
      std::make_pair(BrickFace::kNegY, Eigen::Vector2d(-0.023, 0)));

  auto finger_face_assignments_source = builder.AddSystem<
      systems::ConstantValueSource<double>>(
      Value<std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>>(
          finger_face_assignments));
  builder.Connect(finger_face_assignments_source->get_output_port(0),
                  qp_controller->get_input_port_finger_face_assignments());

  // thetaddot_planned is 0. Use a constant source.
  auto brick_acceleration_planned_source =
      builder.AddSystem<systems::ConstantVectorSource<double>>(
          Eigen::Vector3d::Zero());
  builder.Connect(brick_acceleration_planned_source->get_output_port(),
                  qp_controller->get_input_port_desired_brick_acceleration());

  // The planned theta trajectory is from 0 to 90 degree in 1 second.
  const trajectories::PiecewisePolynomial<double> brick_planned_traj =
      trajectories::PiecewisePolynomial<double>::FirstOrderHold(
          {0, 1}, {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, M_PI_2)});

  auto brick_traj_source = builder.AddSystem<systems::TrajectorySource<double>>(
      brick_planned_traj, 1 /* take 1st derivatives */);
  builder.Connect(brick_traj_source->get_output_port(),
                  qp_controller->get_input_port_desired_state());

  /** Apply zero torque to the joint motor. */
  auto gripper_actuation_source =
      builder.AddSystem<systems::TrajectorySource<double>>(
          trajectories::PiecewisePolynomial<double>::ZeroOrderHold(
              {0, 1}, {Eigen::Matrix<double, 6, 1>::Zero(),
                       Eigen::Matrix<double, 6, 1>::Zero()}));
  builder.Connect(gripper_actuation_source->get_output_port(),
                  planar_gripper->GetInputPort("actuation"));

//  // Log the state.
//  auto signal_logger = builder.AddSystem<systems::SignalLogger<double>>(
//      gripper_brick.plant().num_positions() +
//      gripper_brick.plant().num_velocities());
//  builder.Connect(gripper_brick.get_state_output_port(),
//                  signal_logger->get_input_port());

//  // Create a context for this system:
//  std::unique_ptr<systems::Context<double>> diagram_context =
//      diagram->CreateDefaultContext();
//  diagram->SetDefaultContext(diagram_context.get());
//  systems::Context<double>* plant_context =
//      &(diagram->GetMutableSubsystemContext(gripper_brick.plant(),
//                                            diagram_context.get()));
//
//  // Set initial conditions.
//  // Find a pose such that the fingers are not contacting the brick.
//  Eigen::VectorXd q0(gripper_brick.plant().num_positions());
//  q0.setZero();
//  for (const Finger finger :
//       {Finger::kFinger1, Finger::kFinger2, Finger::kFinger3}) {
//    q0(gripper_brick.finger_base_position_index(finger)) = -1.;
//    q0(gripper_brick.finger_mid_position_index(finger)) = -1.;
//  }
//  gripper_brick.plant().SetPositions(plant_context, q0);

  // Connect drake visualizer.
  geometry::ConnectDrakeVisualizer(
      &builder, planar_gripper->get_mutable_scene_graph(),
      planar_gripper->GetOutputPort("pose_bundle"));

  // TODO(rcory) viz_converter needs to be updated to take the unordered map
  //  output of qp_controller output port control.
//  // To visualize the applied spatial forces.
//  lcm::DrakeLcm drake_lcm;
//  systems::lcm::LcmInterfaceSystem* lcm =
//      builder.AddSystem<systems::lcm::LcmInterfaceSystem>(&drake_lcm);
//  auto viz_converter = builder.AddSystem<ExternalSpatialToSpatialViz>(
//      planar_gripper->get_multibody_plant(), planar_gripper->get_brick_index(),
//      1);
//  builder.Connect(qp_controller->get_output_port_contact_force(),
//                  viz_converter->get_input_port(0));
//  builder.Connect(planar_gripper->GetOutputPort("brick_state"),
//                   viz_converter->get_input_port(1));
//  multibody::ConnectSpatialForcesToDrakeVisualizer(
//      &builder, planar_gripper->get_multibody_plant(),
//      viz_converter->get_output_port(0), lcm);

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
                                   Eigen::Vector2d::Zero());

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);

//  const Eigen::VectorXd sample_times = signal_logger->sample_times();
//  const Eigen::MatrixXd states = signal_logger->data();
//  Eigen::Matrix3Xd brick_planned_samples(3, sample_times.rows());
//  const auto brick_dot_planned_traj = brick_planned_traj.derivative(1);
//  Eigen::Matrix3Xd brick_dot_planned_samples(3, sample_times.rows());
//  for (int i = 0; i < sample_times.rows(); ++i) {
//    brick_planned_samples.col(i) = brick_planned_traj.value(sample_times(i));
//    // The derivative of FirstOrderHold piecewise polynomial is a
//    // ZeroOrderHold
//    // piecewise polynomial, but the thetadot computed after the end of the
//    // trajectory is wrong from thetadot_planned_traj.
//    if (sample_times(i) < brick_planned_traj.end_time()) {
//      brick_dot_planned_samples.col(i) =
//          brick_dot_planned_traj.value(sample_times(i));
//    } else {
//      brick_dot_planned_samples.col(i).setZero();
//    }
//  }
//  Eigen::Matrix<double, 6, Eigen::Dynamic> brick_states(6, sample_times.rows());
//  brick_states.row(0) =
//      states.row(gripper_brick.brick_translate_y_position_index());
//  brick_states.row(1) =
//      states.row(gripper_brick.brick_translate_z_position_index());
//  brick_states.row(2) =
//      states.row(gripper_brick.brick_revolute_x_position_index());
//  brick_states.row(3) =
//      states.row(gripper_brick.plant().num_positions() +
//                 gripper_brick.brick_translate_y_position_index());
//  brick_states.row(4) =
//      states.row(gripper_brick.plant().num_positions() +
//                 gripper_brick.brick_translate_z_position_index());
//  brick_states.row(5) =
//      states.row(gripper_brick.plant().num_positions() +
//                 gripper_brick.brick_revolute_x_position_index());
//
//  common::CallPython("figure");
//  for (int i = 0; i < 3; ++i) {
//    auto ax1 = common::CallPython("subplot", 3, 2, 2 * i + 1);
//    ax1.attr("set_ylabel")("x (m)");
//    common::CallPython("plot", sample_times, brick_states.row(i).transpose(),
//                       common::ToPythonKwargs("label", "sim"));
//    common::CallPython("plot", sample_times,
//                       brick_planned_samples.row(i).transpose(),
//                       common::ToPythonKwargs("label", "plan"));
//    ax1.attr("legend")();
//    auto ax2 = common::CallPython("subplot", 3, 2, 2 * i + 2);
//    common::CallPython("plot", sample_times,
//                       brick_states.row(3 + i).transpose(),
//                       common::ToPythonKwargs("label", "sim"));
//    common::CallPython("plot", sample_times,
//                       brick_dot_planned_samples.row(3 + i).transpose(),
//                       common::ToPythonKwargs("label", "plan"));
//    ax2.attr("legend")();
//    ax2.attr("set_ylabel")("dot (m/s)");
//    ax2.attr("set_xlabel")("time (s)");
//  }
//  common::CallPython("show");
  return 0;
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::planar_gripper::DoMain();
}
