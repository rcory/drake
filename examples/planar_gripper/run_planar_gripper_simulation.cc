/// @file
///
/// This demo simulates a planar-gripper (three two-degree-of-freedom fingers
/// moving in a plane) which manipulates a brick through contact-interactions.
///
/// This simulation can be configured to run in one of two control modes:
/// position control or torque control. In position control mode, desired state
/// is communicated via LCM and fed into a trajectory tracking
/// InverseDynamicsController (within the PlanarGripper diagram). In torque
/// control mode, desired torques are communicated via LCM but are instead
/// directly fed into the actuation input port of the MBP. The control mode can
/// be configured by setting the flag `use_position_control` to true (default)
/// for position control mode, and setting it to false for torque control mode.
///
/// The planar-gripper coordinate frame is illustrated in
/// `planar_gripper_common.h`. Users have the option to either orient the
/// gravity vector to point along the -Gz axis, i.e., simulating the case when
/// the planar-gripper is stood up vertically, or have gravity point along the
/// -Gx axis, i.e., simulating the case when the planar-gripper is laid down
/// flat on the floor.
///
/// To support the vertical case (gravity acting along -Gz) in hardware, we use
/// a plexiglass lid (ceiling) that is opposite the planar-gripper floor in
/// order to keep the brick's motion constrained to the Gy-Gz plane. That is,
/// when the lid is closed the brick is "squeezed" between the ceiling and floor
/// and is also physically constrained along the Gx-axis due to contact with
/// these surfaces. For simulation, we mimic this contact interaction by fixing
/// the amount by which the brick geometry penetrates the floor geometry
/// (without considering the ceiling), and can specify this penetration depth
/// via the flag `brick_floor_penetration'. To enforce zero contact between the
/// brick and floor, set this flag to zero.
///
/// For the horizontal case in hardware, gravity (acting along -Gx) keeps the
/// brick's motion constrained to lie in the Gy-Gz plane (no ceiling required),
/// and therefore the plexiglass lid is left open. This means surface contact
/// only occurs between the brick and the floor. In simulation, we define an
/// additional prismatic degree of freedom for the brick along the Gx axis, such
/// that the brick's position along Gx (i.e., it's contact penetration) is
/// determined by the gravitational and floor contact forces acting on the
/// brick. In this case, the `brick_floor_penetration` flag specifies only the
/// initial brick/floor penetration depth.
///
/// @Note: The keyframes contained in `postures.txt` are strictly for simulating
///        the vertical case with gravity off. Using these keyframes to simulate
///        any other case may cause the simulation to fail.
///
/// Example usage:
///
/// # Terminal 1
/// ./bazel-bin/examples/planar_gripper/run_planar_gripper_trajectory_publisher
///
/// # Terminal 2
/// ./bazel-bin/examples/planar_gripper/run_planar_gripper_simulation

// TODO(rcory) Include a README.md that explains the use cases for this
//  example.

#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/examples/planar_gripper/planar_gripper.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/examples/planar_gripper/planar_gripper_lcm.h"
#include "drake/examples/planar_gripper/planar_gripper_utils.h"
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

namespace drake {
namespace examples {
namespace planar_gripper {
namespace {

using multibody::MultibodyPlant;
using geometry::SceneGraph;

// TODO(rcory) Move all common flags to a shared YAML file.
DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_double(simulation_time, 20,
              "Desired duration of the simulation in seconds.");
DEFINE_double(time_step, 1e-3,
              "If greater than zero, the plant is modeled as a system with "
              "discrete updates and period equal to this time_step. "
              "If 0, the plant is modeled as a continuous system.");
DEFINE_double(penetration_allowance, 1e-3,
              "The contact penetration allowance.");
DEFINE_double(stiction_tolerance, 1e-3, "MBP v_stiction_tolerance");
DEFINE_double(floor_coef_static_friction, 0.5,
              "The floor's coefficient of static friction");
DEFINE_double(floor_coef_kinetic_friction, 0.5,
              "The floor's coefficient of kinetic friction");
DEFINE_double(brick_floor_penetration, 1e-4,
              "Determines how much the brick should penetrate the floor "
              "(in meters). When simulating the vertical case this penetration "
              "distance will remain fixed.");
DEFINE_string(orientation, "vertical",
              "The orientation of the planar gripper. Options are {vertical, "
              "horizontal}.");
DEFINE_bool(visualize_contacts, true,
            "Visualize contacts in Drake visualizer.");
DEFINE_string(brick_type, "planar", "The brick type {pinned, planar}");
DEFINE_bool(
    use_position_control, true,
    "If true (default) we simulate position control via inverse dynamics "
    "control. If false we actuate torques directly.");
DEFINE_string(keyframes_filename, "planar_brick_multi_mode.txt",
              "The name of the file containing the keyframes.");
DEFINE_bool(zero_gravity, true, "Set MBP gravity vector to zero?");
DEFINE_bool(add_floor, true, "Adds a floor to the simulation");

int DoMain() {
  systems::DiagramBuilder<double> builder;
  auto planar_gripper = builder.AddSystem<PlanarGripper>(
      FLAGS_time_step,
      FLAGS_use_position_control ? ControlType::kPosition
                                 : ControlType::kTorque,
      FLAGS_add_floor);

  // Set some plant parameters.
  planar_gripper->set_floor_coef_static_friction(
      FLAGS_floor_coef_static_friction);
  planar_gripper->set_floor_coef_kinetic_friction(
      FLAGS_floor_coef_kinetic_friction);
  planar_gripper->set_brick_floor_penetration(FLAGS_brick_floor_penetration);

  // Setup the pinned or planar brick version of the plant.
  if (FLAGS_brick_type == "pinned") {
    planar_gripper->SetupPinBrick(FLAGS_orientation);
  } else if (FLAGS_brick_type == "planar") {
    planar_gripper->SetupPlanarBrick(FLAGS_orientation);
  } else {
    throw std::runtime_error("Unknown BrickType.");
  }
  planar_gripper->set_penetration_allowance(FLAGS_penetration_allowance);
  planar_gripper->set_stiction_tolerance(FLAGS_stiction_tolerance);
  if (FLAGS_zero_gravity) {
    planar_gripper->zero_gravity();
  }

  // Finalize and build the diagram.
  planar_gripper->Finalize();

  lcm::DrakeLcm drake_lcm;
  systems::lcm::LcmInterfaceSystem* lcm =
      builder.AddSystem<systems::lcm::LcmInterfaceSystem>(&drake_lcm);

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
    auto user_to_plant_state_ordering =
        builder.AddSystem<MapUserOrderedStateToPlantState>(
            planar_gripper->get_control_plant(),
            GetPreferredGripperJointOrdering(),
            planar_gripper->get_planar_gripper_index());
    builder.Connect(command_decoder->get_state_output_port(),
                    user_to_plant_state_ordering->get_input_port(0));
    builder.Connect(user_to_plant_state_ordering->get_output_port(0),
                    planar_gripper->GetInputPort("desired_gripper_state"));
  } else {  // Use torque control.
    auto user_to_plant_actuation_ordering =
        builder.AddSystem<MapUserOrderedActuationToPlantActuation>(
            planar_gripper->get_control_plant(),
            GetPreferredGripperJointOrdering(),
            planar_gripper->get_planar_gripper_index());
    builder.Connect(command_decoder->get_torques_output_port(),
                    user_to_plant_actuation_ordering->get_input_port(0));
    builder.Connect(user_to_plant_actuation_ordering->get_output_port(0),
                    planar_gripper->GetInputPort("torque_control_u"));
  }

  // publish body frames.
  auto frame_viz = builder.AddSystem<FrameViz>(
      planar_gripper->get_multibody_plant(), &drake_lcm, 1.0 / 60.0, false);
  builder.Connect(planar_gripper->GetOutputPort("plant_state"),
                  frame_viz->get_input_port(0));

  geometry::ConnectDrakeVisualizer(&builder,
                                   planar_gripper->get_mutable_scene_graph(),
                                   planar_gripper->GetOutputPort("pose_bundle"),
                                   lcm, geometry::Role::kIllustration);

  // Publish contact results for visualization.
  if (FLAGS_visualize_contacts) {
    ConnectContactResultsToDrakeVisualizer(
        &builder, planar_gripper->get_mutable_multibody_plant(),
        planar_gripper->GetOutputPort("contact_results"));
  }

  // Publish planar gripper status via LCM.
  auto status_pub = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<drake::lcmt_planar_gripper_status>(
          "PLANAR_GRIPPER_STATUS", lcm, get_planar_gripper_lcm_period()));
  auto status_encoder = builder.AddSystem<GripperStatusEncoder>();

  builder.Connect(planar_gripper->GetOutputPort("gripper_state"),
                  status_encoder->get_state_input_port());
  builder.Connect(planar_gripper->GetOutputPort("force_sensor"),
                  status_encoder->get_force_input_port());
  builder.Connect(status_encoder->get_output_port(0),
                  status_pub->get_input_port());

  // Additionally, publish the entire MBP state via LCM.
  const MultibodyPlant<double> &plant = planar_gripper->get_multibody_plant();
  const SceneGraph<double>& scene_graph = planar_gripper->get_scene_graph();
  auto state_pub = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<drake::lcmt_planar_plant_state>(
          "ESTIMATED_PLANT_STATE", lcm, get_planar_gripper_lcm_period()));
  auto estimated_plant_state_enc = builder.AddSystem<QPEstimatedStateEncoder>(
      plant.num_multibody_states());
  builder.Connect(planar_gripper->GetOutputPort("plant_state"),
                  estimated_plant_state_enc->get_input_port(0));
  builder.Connect(
      estimated_plant_state_enc->GetOutputPort("planar_plant_state_lcm"),
      state_pub->get_input_port());

  // Publish the finger face assignments.
  auto finger_face_assignments_pub =
      builder.AddSystem(systems::lcm::LcmPublisherSystem::Make<
                        drake::lcmt_planar_gripper_finger_face_assignments>(
          "FINGER_FACE_ASSIGNMENTS", lcm, get_planar_gripper_lcm_period()));
  auto finger_face_assignments_enc =
      builder.AddSystem<QPFingerFaceAssignmentsEncoder>();
  builder.Connect(
      finger_face_assignments_enc->GetOutputPort("finger_face_assignments_lcm"),
      finger_face_assignments_pub->get_input_port());

  auto finger_face_assigner =
      builder.AddSystem<FingerFaceAssigner>(plant, scene_graph);
  builder.Connect(planar_gripper->GetOutputPort("contact_results"),
                  finger_face_assigner->GetInputPort("contact_results"));
  builder.Connect(planar_gripper->GetOutputPort("scene_graph_query"),
                  finger_face_assigner->GetInputPort("geometry_query"));
  builder.Connect(planar_gripper->GetOutputPort("plant_state"),
                  finger_face_assigner->GetInputPort("plant_state"));
  builder.Connect(
      finger_face_assigner->GetOutputPort("finger_face_assignments"),
      finger_face_assignments_enc->GetInputPort("qp_finger_face_assignments"));

  auto diagram = builder.Build();

  // Extract the initial gripper and brick poses by parsing the keyframe file.
  // The brick's pose consists of {y_position, z_position, x_rotation_angle}.
  const std::string keyframe_path =
      "drake/examples/planar_gripper/keyframes/" + FLAGS_keyframes_filename;
  MatrixX<double> finger_keyframes;
  std::map<std::string, int> finger_joint_name_to_row_index_map;
  std::pair<MatrixX<double>, std::map<std::string, int>> brick_keyframe_info;

  // Note: The keyframe file is parsed strictly for extracting initial
  // conditions. The `time` and `modes` values are unused.
  VectorX<double> times;
  MatrixX<double> modes;
  std::tie(finger_keyframes, finger_joint_name_to_row_index_map) =
      ParseKeyframesAndModes(keyframe_path, &times, &modes,
                             &brick_keyframe_info);
  DRAKE_DEMAND(times.size() == finger_keyframes.cols());
  DRAKE_DEMAND(modes.rows() == 3 && modes.cols() == finger_keyframes.cols());

  finger_keyframes = ReorderKeyframesForPlant(
      planar_gripper->get_control_plant(), finger_keyframes,
      &finger_joint_name_to_row_index_map);

  // Create the initial condition vector. Set initial joint velocities to zero.
  VectorX<double> gripper_initial_positions =
      VectorX<double>::Zero(kNumGripperJoints);
  gripper_initial_positions =
      finger_keyframes.block(0, 0, kNumGripperJoints, 1);

  // Create a context for the diagram.
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& planar_gripper_context =
      diagram->GetMutableSubsystemContext(*planar_gripper,
                                          diagram_context.get());

  planar_gripper->SetGripperPosition(&planar_gripper_context,
                                     gripper_initial_positions);

  std::map<std::string, double> init_brick_pos_map;
  const int rx_index = brick_keyframe_info.second["brick_revolute_x_joint"];
  init_brick_pos_map["brick_revolute_x_joint"] =
      brick_keyframe_info.first(rx_index, 0);
  if (FLAGS_brick_type == "planar") {
    const int ty_index = brick_keyframe_info.second["brick_translate_y_joint"];
    const int tz_index = brick_keyframe_info.second["brick_translate_z_joint"];
    init_brick_pos_map["brick_translate_y_joint"] =
        brick_keyframe_info.first(ty_index, 0);
    init_brick_pos_map["brick_translate_z_joint"] =
        brick_keyframe_info.first(tz_index, 0);
  }
  if (FLAGS_orientation == "horizontal") {
    init_brick_pos_map["brick_translate_x_joint"] = 0;
  }
  auto brick_initial_positions =
      planar_gripper->MakeBrickPositionVector(init_brick_pos_map);
  planar_gripper->SetBrickPosition(&planar_gripper_context,
                                   brick_initial_positions);

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  systems::Context<double>& simulator_context = simulator.get_mutable_context();
  auto Sx = MakeStateSelectorMatrix(planar_gripper->get_control_plant(),
                                    GetPreferredGripperJointOrdering());
  command_decoder->set_initial_position(
      &diagram->GetMutableSubsystemContext(*command_decoder,
                                           &simulator_context),
      Sx.topLeftCorner(Sx.rows() / 2, Sx.cols() / 2) *
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
