#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/common/proto/call_python.h"
#include "drake/common/text_logging_gflags.h"
#include "drake/examples/planar_gripper/brick_qp.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_value_source.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/signal_logger.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/multibody/plant/spatial_forces_to_lcm.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
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

DEFINE_double(yc, 0, "y contact point");
DEFINE_double(zc, 0.046, "z contact point");
DEFINE_double(theta0, -M_PI_4, "initial theta");
DEFINE_double(thetaf, M_PI_4, "final theta");
DEFINE_double(T, 1.0, "time horizon");
DEFINE_double(force_scale, .05, "force viz scale factor");

int DoMain() {
  systems::DiagramBuilder<double> builder;
  geometry::SceneGraph<double>& scene_graph =
      *builder.AddSystem<geometry::SceneGraph<double>>();

  multibody::MultibodyPlant<double>& plant =
      *builder.AddSystem<multibody::MultibodyPlant<double>>(FLAGS_time_step);

  auto plant_id =
      multibody::Parser(&plant, &scene_graph)
          .AddModelFromFile(FindResourceOrThrow(
                                "drake/examples/planar_gripper/1dof_brick.sdf"),
                            "object");
  plant.Finalize();

  // Connect MBP snd SG.
  builder.Connect(
      plant.get_geometry_poses_output_port(),
      scene_graph.get_source_pose_port(plant.get_source_id().value()));
  builder.Connect(scene_graph.get_query_output_port(),
                  plant.get_geometry_query_input_port());

  lcm::DrakeLcm lcm;
  geometry::ConnectDrakeVisualizer(&builder, scene_graph, &lcm);

  // Publish contact results for visualization.
  ConnectContactResultsToDrakeVisualizer(&builder, plant, &lcm);

  // QP controller
  double Kp = 3;
  double Kd = 3;
  double weight_thetaddot_error = 1;
  double weight_f_Cb_B = 1;
  double mu = 0.5;
  double damping =
      plant.GetJointByName<multibody::RevoluteJoint>("brick_pin_joint")
          .damping();
  auto qp_controller = builder.AddSystem<BrickInstantaneousQPController>(
      &plant, Kp, Kd, weight_thetaddot_error, weight_f_Cb_B, mu, damping);

  // Connect the QP controller
  builder.Connect(plant.get_state_output_port(plant_id),
                  qp_controller->get_input_port_estimated_state());
  builder.Connect(qp_controller->get_output_port(0),
                  plant.get_applied_spatial_force_input_port());

  // To visualize the applied spatial forces.
  auto converter = builder.AddSystem<ExternalSpatialToSpatialViz>(
      plant, plant_id, FLAGS_force_scale);
  builder.Connect(qp_controller->get_output_port(0),
                  converter->get_input_port(0));
  multibody::ConnectSpatialForcesToDrakeVisualizer(
      &builder, plant, converter->get_output_port(0), &lcm);
  builder.Connect(plant.get_state_output_port(), converter->get_input_port(1));

  // Always get in contact with the +z face.
  auto contact_face_source =
      builder.AddSystem<systems::ConstantValueSource<double>>(
          Value<BrickFace>(BrickFace::kPosZ));
  builder.Connect(contact_face_source->get_output_port(0),
                  qp_controller->get_input_port_contact_face());

  // Always make contact at position (-0.01, 0.023).
  auto p_BCb_source = builder.AddSystem<systems::ConstantVectorSource<double>>(
      Eigen::Vector2d(FLAGS_yc, FLAGS_zc));
  builder.Connect(p_BCb_source->get_output_port(),
                  qp_controller->get_input_port_p_BCb());

  // thetaddot_planned is 0. Use a constant source.
  auto thetaddot_planned_source =
      builder.AddSystem<systems::ConstantVectorSource<double>>(Vector1d(0));
  builder.Connect(thetaddot_planned_source->get_output_port(),
                  qp_controller->get_input_port_desired_acceleration());

  // The planned theta trajectory is from 0 to 90 degree in 1 second.
  const trajectories::PiecewisePolynomial<double> theta_planned_traj =
      trajectories::PiecewisePolynomial<double>::FirstOrderHold(
          {0, FLAGS_T}, {Vector1d(FLAGS_theta0), Vector1d(FLAGS_thetaf)});

  auto theta_traj_source = builder.AddSystem<systems::TrajectorySource<double>>(
      theta_planned_traj, 1.5 /* take 1st derivatives */);
  builder.Connect(theta_traj_source->get_output_port(),
                  qp_controller->get_input_port_desired_state());

  // Log the state.
  auto signal_logger = builder.AddSystem<systems::SignalLogger<double>>(2);
  builder.Connect(plant.get_state_output_port(),
                  signal_logger->get_input_port());

  auto diagram = builder.Build();

  // Create a context for this system:
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  // Set initial conditions.
  plant.SetPositions(&plant_context, Vector1d(FLAGS_theta0));

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

  // Publish the initial frames
  PublishInitialFrames(plant_context, plant, lcm);

  simulator.set_publish_every_time_step(false);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);

  const Eigen::VectorXd sample_times = signal_logger->sample_times();
  const Eigen::Matrix2Xd brick_states = signal_logger->data();
  common::CallPython("figure");
  auto ax1 = common::CallPython("subplot", 2, 1, 1);
  ax1.attr("set_ylabel")("theta (rad)");
  common::CallPython("plot", sample_times, brick_states.row(0).transpose());
  auto ax2 = common::CallPython("subplot", 2, 1, 2);
  common::CallPython("plot", sample_times, brick_states.row(1).transpose());
  ax2.attr("set_ylabel")("thetadot (rad/s)");
  ax2.attr("set_xlabel")("time (s)");
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
