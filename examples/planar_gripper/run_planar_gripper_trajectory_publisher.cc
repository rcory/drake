/// @file
///
/// Implements a trajectory publisher for the planar gripper simulation by
/// interpolating a sequence of joint poses (keyframes) for the fingers, where
/// each keyframe represents a static-equilibrium condition for the brick (i.e.,
/// the net wrench on the brick is zero). This publisher communicates with the
/// simulator via LCM (publishing a desired state) and generates a sequenced
/// playback of position keyframes at an arbitrarily set speed, configured via
/// the flag `keyframe_dt`.

#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/examples/planar_gripper/planar_gripper_lcm.h"
#include "drake/examples/planar_gripper/planar_manipuland_lcm.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/lcmt_planar_gripper_command.hpp"
#include "drake/lcmt_planar_gripper_status.hpp"
#include "drake/lcmt_planar_manipuland_status.hpp"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/trajectory_source.h"

namespace drake {
namespace examples {
namespace planar_gripper {
namespace {

DEFINE_double(keyframe_dt, 0.1,
              "Defines a uniform time step between `break` points in our "
              "spline interpolator (see "
              "PiecewisePolynomial::CubicShapePreserving), where each keyframe "
              "corresponds to a `knot` point. Note that keyframe data in "
              "postures.txt contains static equilibrium poses and we play "
              "these back at an arbitrary speed for this simulation.");
DEFINE_string(keyframes_filename, "postures.txt",
              "The name of the file containing the keyframes.");

const char* const kLcmGripperStatusChannel = "PLANAR_GRIPPER_STATUS";
const char* const kLcmBrickStatusChannel = "BRICK_STATUS";
const char* const kLcmPlantStateChannel = "PLANAR_PLANT_STATE";
const char* const kLcmGripperCommandChannel = "PLANAR_GRIPPER_COMMAND";

int DoMain() {
  lcm::DrakeLcm lcm;
  systems::DiagramBuilder<double> builder;

  // Create the full plant. Contains the fingers and brick. Used
  // (only) to extract joint velocity ordering.
  const std::string full_name =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_gripper.sdf");
  MultibodyPlant<double> plant(FLAGS_keyframe_dt);
  multibody::Parser(&plant).AddModelFromFile(full_name);
  WeldGripperFrames<double>(&plant);

  // Adds the brick to be manipulated.
  const std::string brick_file_name =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_brick.sdf");
  auto brick_index =
      multibody::Parser(&plant).AddModelFromFile(brick_file_name, "brick");

  // When the planar-gripper is welded via WeldGripperFrames(), motion always
  // lies in the world Y-Z plane (because the planar-gripper frame is aligned
  // with the world frame).
  const multibody::Frame<double>& brick_base_frame =
      plant.GetFrameByName("brick_base_link", brick_index);
  plant.WeldFrames(plant.world_frame(), brick_base_frame);

  plant.Finalize();

  // Create the controlled plant. Contains only the fingers (no brick). Used
  // (only) to extract joint velocity ordering.
  MultibodyPlant<double> control_plant(FLAGS_keyframe_dt);
  multibody::Parser(&control_plant).AddModelFromFile(full_name);
  WeldGripperFrames<double>(&control_plant);
  control_plant.Finalize();

  const int num_fingers = 3;
  const int num_joints = num_fingers * 2;

  // We'll directly fix the input to the status receiver later from our lcm
  // subscriber.
  auto gripper_status_decoder = builder.AddSystem<GripperStatusDecoder>();

  // Communicates the gripper command.
  auto gripper_command_pub = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<lcmt_planar_gripper_command>(
          kLcmGripperCommandChannel, &lcm));
  auto gripper_command_encoder = builder.AddSystem<GripperCommandEncoder>();

  // Communicates the pose of the brick.
  auto brick_status_pub = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<lcmt_planar_manipuland_status>(
          kLcmBrickStatusChannel, &lcm));
  auto brick_status_encoder =
      builder.AddSystem<PlanarManipulandStatusEncoder>();

  // Communicates the plant state.
  auto plant_state_pub = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<lcmt_planar_plant_state>(
          kLcmPlantStateChannel, &lcm));
  auto plant_state_encoder =
      builder.AddSystem<QPEstimatedStateEncoder>(plant.num_multibody_states());

  // Parse the keyframes from a file.
  const std::string keyframe_path =
      "drake/examples/planar_gripper/" + FLAGS_keyframes_filename;
  MatrixX<double> finger_keyframes;
  std::map<std::string, int> finger_joint_name_to_row_index_map;
  std::pair<MatrixX<double>, std::map<std::string, int>> brick_keyframe_info;
  std::tie(finger_keyframes, finger_joint_name_to_row_index_map) =
      ParseKeyframes(keyframe_path, &brick_keyframe_info);
  finger_keyframes = ReorderKeyframesForPlant(control_plant, finger_keyframes,
                                       &finger_joint_name_to_row_index_map);
  // Here we assume the gripper frame G is aligned with the world frame W, e.g.,
  // as given by calling WeldGripperFrames().
  // TODO(rcory) Figure out a way to enforce this.

  // Creates the time vector for the plan interpolator.
  Eigen::VectorXd times = Eigen::VectorXd::Zero(finger_keyframes.cols());
  for (int i = 1; i < finger_keyframes.cols(); ++i) {
    times(i) = i * FLAGS_keyframe_dt;
  }
  const auto finger_pp =
      trajectories::PiecewisePolynomial<double>::CubicShapePreserving(
          times, finger_keyframes);
  auto finger_state_src = builder.AddSystem<systems::TrajectorySource<double>>(
      finger_pp, 1 /* with one derivative */);

  MatrixX<double> brick_keyframes = brick_keyframe_info.first;
  std::map<std::string, int> brick_joint_name_to_row_index_map =
      brick_keyframe_info.second;
  const auto brick_pp =
      trajectories::PiecewisePolynomial<double>::CubicShapePreserving(
          times, brick_keyframes);
  auto brick_state_src = builder.AddSystem<systems::TrajectorySource<double>>(
      brick_pp, 1 /* with one derivative */);

  // Concatenate the finger and brick keyframes in order to publish the
  // entire plant position vector.
  DRAKE_DEMAND(finger_keyframes.cols() == brick_keyframes.cols());
  MatrixX<double> plant_keyframes(
      finger_keyframes.rows() + brick_keyframes.rows(),
      finger_keyframes.cols());
  plant_keyframes << finger_keyframes, brick_keyframes;
  std::map<std::string, int> plant_joint_name_to_row_index_map;
  plant_joint_name_to_row_index_map.insert(
      finger_joint_name_to_row_index_map.begin(),
      finger_joint_name_to_row_index_map.end());
  for (auto& iter : brick_joint_name_to_row_index_map) {
    iter.second += finger_joint_name_to_row_index_map.size();
  }
  plant_joint_name_to_row_index_map.insert(
      brick_joint_name_to_row_index_map.begin(),
      brick_joint_name_to_row_index_map.end());
  plant_keyframes = ReorderKeyframesForPlant(
      plant, plant_keyframes, &plant_joint_name_to_row_index_map);
  const auto plant_pp =
      trajectories::PiecewisePolynomial<double>::CubicShapePreserving(
          times, plant_keyframes);
  auto plant_state_src = builder.AddSystem<systems::TrajectorySource<double>>(
      plant_pp, 1 /* with one derivative */);

  VectorX<double> torques(num_joints); torques.setZero();
  auto torques_src = builder.AddSystem<systems::ConstantVectorSource>(torques);
  builder.Connect(finger_state_src->get_output_port(),
                  gripper_command_encoder->get_state_input_port());
  builder.Connect(torques_src->get_output_port(),
                  gripper_command_encoder->get_torques_input_port());
  builder.Connect(gripper_command_encoder->get_output_port(0),
                  gripper_command_pub->get_input_port());

  builder.Connect(brick_state_src->get_output_port(),
                  brick_status_encoder->get_input_port(0));
  builder.Connect(brick_status_encoder->get_output_port(0),
                  brick_status_pub->get_input_port());

  builder.Connect(plant_state_src->get_output_port(),
                  plant_state_encoder->get_input_port(0));
  builder.Connect(plant_state_encoder->get_output_port(0),
                  plant_state_pub->get_input_port());

  auto owned_diagram = builder.Build();
  const systems::Diagram<double>* diagram = owned_diagram.get();
  systems::Simulator<double> simulator(std::move(owned_diagram));

  // Wait for the first message.
  drake::log()->info("Waiting for first lcmt_planar_gripper_status");
  lcm::Subscriber<lcmt_planar_gripper_status> status_sub(
      &lcm, kLcmGripperStatusChannel);
  LcmHandleSubscriptionsUntil(&lcm, [&]() { return status_sub.count() > 0; });

  const lcmt_planar_gripper_status& first_status = status_sub.message();
  DRAKE_DEMAND(first_status.num_fingers == 0 ||
               first_status.num_fingers == num_fingers);

  VectorX<double> q0 = VectorX<double>::Zero(num_joints);
  for (int i = 0; i < first_status.num_fingers; ++i) {
    int st_index = i * 2;
    q0(st_index) = first_status.finger_status[i].joint_position[0];
    q0(st_index + 1) = first_status.finger_status[i].joint_position[1];
  }

  systems::Context<double>& diagram_context = simulator.get_mutable_context();
  const double t0 = first_status.utime * 1e-6;
  diagram_context.SetTime(t0);

  systems::Context<double>& status_context =
      diagram->GetMutableSubsystemContext(*gripper_status_decoder,
                                          &diagram_context);
  auto& status_value = gripper_status_decoder->get_input_port(0).FixValue(
      &status_context, first_status);

  // Run forever, using the lcmt_planar_gripper_status message to dictate when
  // simulation time advances.
  drake::log()->info("Trajectory publisher started.");
  while (true) {
    // Wait for an lcmt_planar_gripper_status message.
    status_sub.clear();
    LcmHandleSubscriptionsUntil(&lcm, [&]() { return status_sub.count() > 0; });
    // Write the lcmt_planar_gripper_status message into the context and
    // advance.
    status_value.GetMutableData()->set_value(status_sub.message());
    const double time = status_sub.message().utime * 1e-6;
    simulator.AdvanceTo(time);
    // Force-publish the lcmt_planar_gripper_command (via the command_pub system
    // within the diagram).
    diagram->Publish(diagram_context);
  }

  // We should never reach here.
  return EXIT_FAILURE;
}

}  // namespace
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::planar_gripper::DoMain();
}
