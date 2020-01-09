#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/common/proto/call_python.h"
#include "drake/common/text_logging_gflags.h"
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
int DoMain() {
  systems::DiagramBuilder<double> builder;

  GripperBrickHelper<double> gripper_brick;

  builder.AddSystem(gripper_brick.owned_diagram());

  // QP controller
  const Eigen::Matrix2d Kp1 = Eigen::Vector2d(10, 10).asDiagonal();
  const Eigen::Matrix2d Kd1 = Eigen::Vector2d(10, 10).asDiagonal();
  double Kp2 = 10;
  double Kd2 = 10;
  double weight_a = 10;
  double weight_thetaddot_error = 1;
  double weight_f_Cb_B = 0.01;
  auto qp_controller = builder.AddSystem<InstantaneousContactForceQPController>(
      &gripper_brick, Kp1, Kd1, Kp2, Kd2, weight_a, weight_thetaddot_error,
      weight_f_Cb_B);

  // Connect the QP controller
  builder.Connect(gripper_brick.get_state_output_port(),
                  qp_controller->get_input_port_estimated_state());
  builder.Connect(qp_controller->get_output_port_contact_force(),
                  gripper_brick.get_applied_spatial_force_input_port());

  // Finger 1 in contact with -z face, Finger 2 in contact with +y face.
  std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>
      finger_face_assignment;
  finger_face_assignment.emplace(
      Finger::kFinger1,
      std::make_pair(BrickFace::kNegZ, Eigen::Vector2d(0, -0.023)));
  finger_face_assignment.emplace(
      Finger::kFinger2,
      std::make_pair(BrickFace::kPosY, Eigen::Vector2d(0.023, 0)));
  finger_face_assignment.emplace(
      Finger::kFinger3,
      std::make_pair(BrickFace::kNegY, Eigen::Vector2d(-0.023, 0)));
  auto contact_face_source = builder.AddSystem<
      systems::ConstantValueSource<double>>(
      Value<std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>>(
          finger_face_assignment));
  builder.Connect(contact_face_source->get_output_port(0),
                  qp_controller->get_input_port_finger_contact());

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
                  gripper_brick.get_actuation_input_port());

  // Log the state.
  auto signal_logger = builder.AddSystem<systems::SignalLogger<double>>(
      gripper_brick.plant().num_positions() +
      gripper_brick.plant().num_velocities());
  builder.Connect(gripper_brick.get_state_output_port(),
                  signal_logger->get_input_port());

  auto diagram = builder.Build();

  // Create a context for this system:
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());
  systems::Context<double>* plant_context =
      &(diagram->GetMutableSubsystemContext(gripper_brick.plant(),
                                            diagram_context.get()));

  // Set initial conditions.
  // Find a pose such that the fingers are not contacting the brick.
  Eigen::VectorXd q0(gripper_brick.plant().num_positions());
  q0.setZero();
  for (const Finger finger :
       {Finger::kFinger1, Finger::kFinger2, Finger::kFinger3}) {
    q0(gripper_brick.finger_base_position_index(finger)) = -1.;
    q0(gripper_brick.finger_mid_position_index(finger)) = -1.;
  }
  gripper_brick.plant().SetPositions(plant_context, q0);

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

  simulator.set_publish_every_time_step(false);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);

  const Eigen::VectorXd sample_times = signal_logger->sample_times();
  const Eigen::MatrixXd states = signal_logger->data();
  Eigen::Matrix3Xd brick_planned_samples(3, sample_times.rows());
  const auto brick_dot_planned_traj = brick_planned_traj.derivative(1);
  Eigen::Matrix3Xd brick_dot_planned_samples(3, sample_times.rows());
  for (int i = 0; i < sample_times.rows(); ++i) {
    brick_planned_samples.col(i) = brick_planned_traj.value(sample_times(i));
    // The derivative of FirstOrderHold piecewise polynomial is a
    // ZeroOrderHold
    // piecewise polynomial, but the thetadot computed after the end of the
    // trajectory is wrong from thetadot_planned_traj.
    if (sample_times(i) < brick_planned_traj.end_time()) {
      brick_dot_planned_samples.col(i) =
          brick_dot_planned_traj.value(sample_times(i));
    } else {
      brick_dot_planned_samples.col(i).setZero();
    }
  }
  Eigen::Matrix<double, 6, Eigen::Dynamic> brick_states(6, sample_times.rows());
  brick_states.row(0) =
      states.row(gripper_brick.brick_translate_y_position_index());
  brick_states.row(1) =
      states.row(gripper_brick.brick_translate_z_position_index());
  brick_states.row(2) =
      states.row(gripper_brick.brick_revolute_x_position_index());
  brick_states.row(3) =
      states.row(gripper_brick.plant().num_positions() +
                 gripper_brick.brick_translate_y_position_index());
  brick_states.row(4) =
      states.row(gripper_brick.plant().num_positions() +
                 gripper_brick.brick_translate_z_position_index());
  brick_states.row(5) =
      states.row(gripper_brick.plant().num_positions() +
                 gripper_brick.brick_revolute_x_position_index());

  common::CallPython("figure");
  for (int i = 0; i < 3; ++i) {
    auto ax1 = common::CallPython("subplot", 3, 2, 2 * i + 1);
    ax1.attr("set_ylabel")("x (m)");
    common::CallPython("plot", sample_times, brick_states.row(i).transpose(),
                       common::ToPythonKwargs("label", "sim"));
    common::CallPython("plot", sample_times,
                       brick_planned_samples.row(i).transpose(),
                       common::ToPythonKwargs("label", "plan"));
    ax1.attr("legend")();
    auto ax2 = common::CallPython("subplot", 3, 2, 2 * i + 2);
    common::CallPython("plot", sample_times,
                       brick_states.row(3 + i).transpose(),
                       common::ToPythonKwargs("label", "sim"));
    common::CallPython("plot", sample_times,
                       brick_dot_planned_samples.row(3 + i).transpose(),
                       common::ToPythonKwargs("label", "plan"));
    ax2.attr("legend")();
    ax2.attr("set_ylabel")("dot (m/s)");
    ax2.attr("set_xlabel")("time (s)");
  }
  common::CallPython("show");
  return 0;
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::logging::HandleSpdlogGflags();
  return drake::examples::planar_gripper::DoMain();
}
