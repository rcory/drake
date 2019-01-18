#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/examples/planar_gripper/planar_gripper_lcm.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/examples/planar_gripper/planar_gripper.h"

namespace drake {
namespace examples {
namespace planar_gripper {
namespace {

DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_double(simulation_time, 4.5,
              "Desired duration of the simulation in seconds.");
DEFINE_double(time_step, 1e-3,
              "If greater than zero, the plant is modeled as a system with "
              "discrete updates and period equal to this time_step. "
              "If 0, the plant is modeled as a continuous system.");
DEFINE_double(penetration_allowance, 1e-3,
              "The contact penetration allowance.");
DEFINE_double(floor_coef_static_friction, 0.5,
              "The floor's coefficient of static friction");
DEFINE_double(floor_coef_kinetic_friction, 0.5,
              "The floor's coefficient of kinetic friction");
DEFINE_double(brick_floor_penetration, 1e-5,
              "Determines how much the brick should penetrate the floor "
              "(in meters). When simulating the vertical case this penetration "
              "distance will remain fixed.");
DEFINE_string(orientation, "horizontal",
              "The orientation of the planar gripper. Options are {vertical, "
              "horizontal}.");
DEFINE_bool(visualize_contacts, true,
            "Visualize contacts in Drake visualizer.");
DEFINE_bool(
    use_position_control, true,
    "If true (default) we simulate position control via inverse dynamics "
    "control. If false we actuate torques directly.");

DEFINE_string(keyframes_filename, "postures_horizontal.txt",
              "The name of the file containing the keyframes.");

int DoMain() {
  systems::DiagramBuilder<double> builder;
  auto planar_gripper = builder.AddSystem<PlanarGripper>(
      FLAGS_time_step, FLAGS_use_position_control);

  // Set some plant parameters.
  planar_gripper->set_floor_coef_static_friction(
      FLAGS_floor_coef_static_friction);
  planar_gripper->set_floor_coef_kinetic_friction(
      FLAGS_floor_coef_kinetic_friction);
  planar_gripper->set_brick_floor_penetration(FLAGS_brick_floor_penetration);

  // Setup the planar brick version of the plant.
  planar_gripper->SetupPlanarBrick(FLAGS_orientation);
  planar_gripper->set_penetration_allowance(FLAGS_penetration_allowance);

  // Finalize and build the diagram.
  planar_gripper->Finalize();

  systems::lcm::LcmInterfaceSystem* lcm =
      builder.AddSystem<systems::lcm::LcmInterfaceSystem>();

  auto command_sub = builder.AddSystem(
      systems::lcm::LcmSubscriberSystem::Make<
          drake::lcmt_planar_gripper_command>("PLANAR_GRIPPER_COMMAND", lcm));
  auto command_decoder = builder.AddSystem<GripperCommandDecoder>();
  builder.Connect(command_sub->get_output_port(),
                  command_decoder->get_input_port(0));

  // The planar gripper "command" LCM message contains entries for both desired
  // state and desired torque. However, the boolean gflag `use_position_control`
  // ultimately controls whether the diagram is wired for position control mode
  // (desired torques are ignored) or torque control mode (desired state is
  // ignored).
  if (FLAGS_use_position_control) {
    builder.Connect(command_decoder->get_state_output_port(),
                    planar_gripper->GetInputPort("desired_gripper_state"));
  } else {  // Use torque control.
    builder.Connect(command_decoder->get_torques_output_port(),
                    planar_gripper->GetInputPort("actuation"));
  }

  geometry::ConnectDrakeVisualizer(
      &builder, planar_gripper->get_mutable_scene_graph(),
      planar_gripper->GetOutputPort("pose_bundle"));

  // Publish contact results for visualization.
  if (FLAGS_visualize_contacts) {
    ConnectContactResultsToDrakeVisualizer(
        &builder, planar_gripper->get_mutable_multibody_plant(),
        planar_gripper->GetOutputPort("contact_results"));
  }

  // Publish planar gripper status via LCM.
  auto status_pub = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<drake::lcmt_planar_gripper_status>(
          "PLANAR_GRIPPER_STATUS", lcm, kGripperLcmStatusPeriod));
  auto status_encoder = builder.AddSystem<GripperStatusEncoder>();

  builder.Connect(planar_gripper->GetOutputPort("gripper_state"),
                  status_encoder->get_state_input_port());
  builder.Connect(planar_gripper->GetOutputPort("force_sensor"),
                  status_encoder->get_force_input_port());
  builder.Connect(status_encoder->get_output_port(0),
                  status_pub->get_input_port());

  auto diagram = builder.Build();

  // Extract the initial gripper and brick poses by parsing the keyframe file.
  // The brick's pose consists of {y_position, z_position, x_rotation_angle}.
  const std::string keyframe_path =
      "drake/examples/planar_gripper/" + FLAGS_keyframes_filename;
  MatrixX<double> keyframes;
  std::map<std::string, int> finger_joint_name_to_row_index_map;
  Vector3<double> brick_initial_2D_pose_G;
  std::tie(keyframes, finger_joint_name_to_row_index_map) =
      ParseKeyframes(keyframe_path, &brick_initial_2D_pose_G);
  keyframes =
      ReorderKeyframesForPlant(planar_gripper->get_control_plant(), keyframes,
                               &finger_joint_name_to_row_index_map);

  // Create the initial condition vector. Set initial joint velocities to zero.
  VectorX<double> gripper_initial_positions =
      VectorX<double>::Zero(kNumGripperJoints);
  gripper_initial_positions =
      keyframes.block(0, 0, kNumGripperJoints, 1);

  // Create a context for the diagram.
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& planar_gripper_context =
      diagram->GetMutableSubsystemContext(*planar_gripper,
                                          diagram_context.get());

  planar_gripper->SetGripperPosition(&planar_gripper_context,
                                     gripper_initial_positions);

  std::map<std::string, double> init_brick_pos_map;
  init_brick_pos_map["brick_translate_y_joint"] = brick_initial_2D_pose_G(0);
  init_brick_pos_map["brick_translate_z_joint"] = brick_initial_2D_pose_G(1);
  init_brick_pos_map["brick_revolute_x_joint"] = brick_initial_2D_pose_G(2);
  if (FLAGS_orientation == "horizontal") {
    init_brick_pos_map["brick_translate_x_joint"] = 0;
  }
  auto brick_initial_positions =
      planar_gripper->MakeBrickPositionVector(init_brick_pos_map);
  planar_gripper->SetBrickPosition(&planar_gripper_context,
                                   brick_initial_positions);

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  systems::Context<double>& simulator_context = simulator.get_mutable_context();
  command_decoder->set_initial_position(
      &diagram->GetMutableSubsystemContext(*command_decoder,
                                           &simulator_context),
      gripper_initial_positions);

  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);

  return 0;
}

}  // namespace
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("A simple planar gripper example.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::planar_gripper::DoMain();
}
