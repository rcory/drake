#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/examples/planar_gripper/planar_gripper_lcm.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/rendering/multibody_position_to_geometry_pose.h"

namespace drake {
namespace examples {
namespace planar_gripper {
namespace {

using geometry::SceneGraph;
using multibody::ModelInstanceIndex;
using multibody::Parser;
using multibody::PrismaticJoint;

DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_double(simulation_time, 20,
              "Desired duration of the simulation in seconds.");

/// A system that takes in a geometry::FramePoseVector and produces a vector
/// of RigidTransforms for all bodies specified at construction. This system
/// facilitates visualization of body frames during playback.
class FramePoseVectorToTransforms final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FramePoseVectorToTransforms)

  FramePoseVectorToTransforms(
      const MultibodyPlant<double>& plant,
      const geometry::SceneGraph<double>& scene_graph,
      const std::vector<std::string>& body_names,
      const std::vector<std::string>& geometry_names,
      const std::vector<geometry::Role>& geometry_roles) {
    DRAKE_DEMAND(body_names.size() == geometry_names.size());
    DRAKE_DEMAND(geometry_names.size() == geometry_roles.size());
    geometry_frame_ids_.resize(geometry_roles.size());
    for (size_t i = 0; i < body_names.size(); i++) {
      geometry::GeometryId geom_id =
          scene_graph.model_inspector().GetGeometryIdByName(
              plant.GetBodyFrameIdOrThrow(
                  plant.GetBodyByName(body_names[i]).index()),
              geometry_roles[i], geometry_names[i]);
      geometry_frame_ids_[i] =
          scene_graph.model_inspector().GetFrameId(geom_id);
    }
    this->DeclareAbstractInputPort("frame_pose_vector",
                                   Value<geometry::FramePoseVector<double>>());
    // Declares an abstract input port of type FramePoseVector (i.e., the
    // output type of MultibodyPositionToGeometryPose object).
    this->DeclareAbstractOutputPort("poses",
                                    &FramePoseVectorToTransforms::CalcOutput);
  }

  void CalcOutput(
      const systems::Context<double>& context,
      std::vector<math::RigidTransform<double>>* X_WBrick_vec) const {
    auto frame_pose_vector =
        this->GetInputPort("frame_pose_vector")
            .Eval<geometry::FramePoseVector<double>>(context);
    X_WBrick_vec->clear();
    for (auto iter : geometry_frame_ids_) {
      X_WBrick_vec->push_back(frame_pose_vector.value(iter));
    }
  }

 private:
  std::vector<drake::geometry::FrameId> geometry_frame_ids_;
};

int DoMain() {
  systems::DiagramBuilder<double> builder;

  SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
  scene_graph.set_name("scene_graph");

  // Make and add the planar_gripper model.
  const std::string full_name =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_gripper.sdf");
  // Setup the MBP with an arbitrary time step to suppress joint limit warnings.
  MultibodyPlant<double> plant(1.0);
  plant.RegisterAsSourceForSceneGraph(&scene_graph);
  auto gripper_model_index =
      Parser(&plant, &scene_graph).AddModelFromFile(full_name);
  WeldGripperFrames<double>(&plant);

  // Adds the brick to be manipulated.
  const std::string brick_file_name =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_brick.sdf");
  const ModelInstanceIndex brick_index =
      Parser(&plant).AddModelFromFile(brick_file_name, "brick");

  // When the planar-gripper is welded via WeldGripperFrames(), motion always
  // lies in the world Y-Z plane (because the planar-gripper frame is aligned
  // with the world frame).
  const multibody::Frame<double>& brick_base_frame =
      plant.GetFrameByName("brick_base_link", brick_index);
  plant.WeldFrames(plant.world_frame(), brick_base_frame);

  plant.Finalize();

  lcm::DrakeLcm drake_lcm;
  systems::lcm::LcmInterfaceSystem* lcm =
      builder.AddSystem<systems::lcm::LcmInterfaceSystem>(&drake_lcm);

  auto plant_state_sub = builder.AddSystem(
      systems::lcm::LcmSubscriberSystem::Make<
          drake::lcmt_planar_plant_state>("PLANAR_PLANT_STATE", lcm));
  auto plant_state_decoder =
      builder.AddSystem<QPEstimatedStateDecoder>(plant.num_multibody_states());
  builder.Connect(plant_state_sub->get_output_port(),
                  plant_state_decoder->get_input_port(0));

  auto pos2geom_sys =
      builder.AddSystem<systems::rendering::MultibodyPositionToGeometryPose>(
          plant, true /* input is MBP state */);
  builder.Connect(plant_state_decoder->get_output_port(0),
                  pos2geom_sys->get_input_port());
  builder.Connect(
      pos2geom_sys->get_output_port(),
      scene_graph.get_source_pose_port(plant.get_source_id().value()));

  // Publish geometry frames.
  auto frame_viz =
      builder.AddSystem<FrameViz>(plant, &drake_lcm, 1.0 / 60.0, true);
  std::vector<std::string> body_names = {"brick_link"};
  std::vector<std::string> geometry_names = {"brick::box_collision"};
  std::vector<geometry::Role> geometry_roles = {geometry::Role::kProximity};
  auto bundle2brick = builder.AddSystem<FramePoseVectorToTransforms>(
      plant, scene_graph, body_names, geometry_names, geometry_roles);
  builder.Connect(pos2geom_sys->GetOutputPort("geometry_pose"),
                  bundle2brick->GetInputPort("frame_pose_vector"));
  builder.Connect(bundle2brick->GetOutputPort("poses"),
                  frame_viz->GetInputPort("poses"));

  geometry::ConnectDrakeVisualizer(&builder, scene_graph, &drake_lcm);

  // Publish planar gripper status via LCM.
  auto status_pub = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<drake::lcmt_planar_gripper_status>(
          "PLANAR_GRIPPER_STATUS", lcm, kGripperLcmPeriod));
  auto status_encoder = builder.AddSystem<GripperStatusEncoder>();

  // Publish a dummy state in order to properly clock the trajectory publisher.
  const int kNumStates = plant.num_multibody_states(gripper_model_index);
  VectorX<double> zero_gripper_state = VectorX<double>::Zero(kNumStates);
  auto zero_gripper_state_src =
      builder.AddSystem<systems::ConstantVectorSource>(zero_gripper_state);
  builder.Connect(zero_gripper_state_src->get_output_port(),
                  status_encoder->get_state_input_port());
  const int kNumSensors = 3;  // For 3 fingers.
  VectorX<double> zero_sensors = VectorX<double>::Zero(2* kNumSensors);
  auto constant_sensor_src =
      builder.AddSystem<systems::ConstantVectorSource>(zero_sensors);
  builder.Connect(constant_sensor_src->get_output_port(),
                  status_encoder->get_force_input_port());
  builder.Connect(status_encoder->get_output_port(0),
                  status_pub->get_input_port());

  auto diagram = builder.Build();

  // Create a context for the diagram.
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

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

