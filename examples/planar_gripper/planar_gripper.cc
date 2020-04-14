#include "drake/examples/planar_gripper/planar_gripper.h"

#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/examples/planar_gripper/planar_gripper_lcm.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace examples {
namespace planar_gripper {

using Eigen::Vector3d;
using geometry::SceneGraph;
using multibody::JointActuatorIndex;
using multibody::ModelInstanceIndex;
using multibody::MultibodyPlant;
using multibody::Parser;
using multibody::PrismaticJoint;
using multibody::RevoluteJoint;

/// Adds a floor to the simulation, modeled as a thin cylinder.
void PlanarGripper::AddFloor(MultibodyPlant<double>* plant,
                             const SceneGraph<double>& scene_graph) {
  // Get info for the brick from the SceneGraph inspector. This is used to
  // determine placement of the floor in order to achieve the specified
  // brick/floor penetration.
  const geometry::SceneGraphInspector<double>& inspector =
      scene_graph.model_inspector();

  // The brick model includes four small sphere collisions at the bottom four
  // corners of the box collision. These four spheres (and not the box) are
  // intended to make contact with the floor. Here we extract the height of
  // these spheres in order to weld the floor at the appropriate height, such
  // that the initial box/floor penetration is given by the flag
  // brick_floor_penetration.
  const geometry::Shape& sphere_shape =
      inspector.GetShape(inspector.GetGeometryIdByName(
          plant->GetBodyFrameIdOrThrow(
              plant->GetBodyByName("brick_link").index()),
          geometry::Role::kProximity, "brick::sphere1_collision"));
  const double sphere_radius =
      dynamic_cast<const geometry::Sphere&>(sphere_shape).radius();
  const math::RigidTransformd X_WS =
      inspector.GetPoseInFrame(inspector.GetGeometryIdByName(
          plant->GetBodyFrameIdOrThrow(
              plant->GetBodyByName("brick_link").index()),
          geometry::Role::kProximity, "brick::sphere1_collision"));

  const double kFloorHeight = 0.001;
  const double kSphereTipXOffset = X_WS.translation()(0) - sphere_radius;
  const drake::multibody::CoulombFriction<double> coef_friction_floor(
      floor_coef_static_friction_, floor_coef_kinetic_friction_);
  const math::RigidTransformd X_WF(
      Eigen::AngleAxisd(M_PI_2, Vector3d::UnitY()),
      Vector3d(
          kSphereTipXOffset - (kFloorHeight / 2.0) + brick_floor_penetration_,
          0, 0));
  //  const Vector4<double> black(0.2, 0.2, 0.2, 1.0);
  const Vector4<double> white(0.8, 0.8, 0.8, 0.6);
  plant->RegisterVisualGeometry(plant->world_body(), X_WF,
                                geometry::Cylinder(.125, kFloorHeight),
                                "FloorVisualGeometry", white);
  plant->RegisterCollisionGeometry(
      plant->world_body(), X_WF, geometry::Cylinder(.125, kFloorHeight),
      "FloorCollisionGeometry", coef_friction_floor);
}

geometry::GeometryId PlanarGripper::fingertip_sphere_geometry_id(
    Finger finger) const {
  return fingertip_sphere_geometry_ids_.at(finger);
}

geometry::GeometryId PlanarGripper::brick_geometry_id() const {
  return brick_geometry_id_;
}

/// Reorders the generalized force output vector of the ID controller
/// (internally using a control plant with only the gripper) to match the
/// actuation input ordering for the full simulation plant (containing gripper
/// and brick).
class GeneralizedForceToActuationOrdering : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GeneralizedForceToActuationOrdering);
  explicit GeneralizedForceToActuationOrdering(
      const MultibodyPlant<double>& plant)
      : Binv_(plant.MakeActuationMatrix().inverse()) {
    this->DeclareVectorInputPort(
        "tau", systems::BasicVector<double>(plant.num_actuators()));
    this->DeclareVectorOutputPort(
        "u", systems::BasicVector<double>(plant.num_actuators()),
        &GeneralizedForceToActuationOrdering::remap_output);
  }

  void remap_output(const systems::Context<double>& context,
                    systems::BasicVector<double>* output_vector) const {
    auto output_value = output_vector->get_mutable_value();
    auto input_value = this->EvalVectorInput(context, 0)->get_value();

    output_value.setZero();
    output_value = Binv_ * input_value;
  }

 private:
  const MatrixX<double> Binv_;
};

/// A system whose input port takes in MBP joint reaction forces and whose
/// outputs correspond to the (planar-only) forces felt at the force sensor,
/// for all three fingers (i.e., fy and fz). Because the task is planar, we
/// ignore any forces/torques not acting in the y-z plane.
class ForceSensorEvaluator : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ForceSensorEvaluator);
  explicit ForceSensorEvaluator(const MultibodyPlant<double>& plant) {
    const int num_sensors = 3;
    for (int i = 1; i <= num_sensors; i++) {
      std::string joint_name =
          "finger" + std::to_string(i) + "_sensor_weldjoint";
      sensor_joint_indices_.push_back(
          plant.GetJointByName<multibody::WeldJoint>(joint_name).index());
    }
    this->DeclareAbstractInputPort(
            "spatial_forces_in",
            Value<std::vector<multibody::SpatialForce<double>>>())
        .get_index();
    this->DeclareVectorOutputPort("force_sensors_out",
                                  systems::BasicVector<double>(num_sensors * 2),
                                  &ForceSensorEvaluator::SetOutput)
        .get_index();
  }

  void SetOutput(const drake::systems::Context<double>& context,
                 drake::systems::BasicVector<double>* output) const {
    const std::vector<multibody::SpatialForce<double>>& spatial_vec =
        this->get_input_port(0)
            .Eval<std::vector<multibody::SpatialForce<double>>>(context);
    auto output_value = output->get_mutable_value();
    // Force sensor (fy, fz) values, measured in the "tip_link" frame.
    output_value.head<2>() =
        spatial_vec[sensor_joint_indices_[0]].translational().tail(2);
    output_value.segment<2>(2) =
        spatial_vec[sensor_joint_indices_[1]].translational().tail(2);
    output_value.tail<2>() =
        spatial_vec[sensor_joint_indices_[2]].translational().tail(2);
  }

 private:
  std::vector<multibody::JointIndex> sensor_joint_indices_;
};

VectorX<double> PlanarGripper::MakeGripperPositionVector(
    const std::map<std::string, double>& map_in) {
  const int kNumGripperPositions = get_num_gripper_positions();
  if (kNumGripperJoints != kNumGripperPositions) {
    throw std::runtime_error(
        "kNumGripperJoints does not match number of positions in "
        "PlanarGripper's plant.");
  }
  if (static_cast<int>(map_in.size()) != kNumGripperPositions) {
    throw std::runtime_error(
        "The number initial condition joints must match the number of "
        "planar-gripper "
        "joints");
  }
  return MakePositionVector(map_in, kNumGripperPositions);
}

VectorX<double> PlanarGripper::GetGripperPosition(
    const systems::Context<double>& diagram_context) const {
  const auto& plant_context =
      this->GetSubsystemContext(*plant_, diagram_context);
  return plant_->GetPositions(plant_context, gripper_index_);
}

VectorX<double> PlanarGripper::GetGripperVelocity(
    const systems::Context<double>& station_context) const {
  const auto& plant_context =
      this->GetSubsystemContext(*plant_, station_context);
  return plant_->GetVelocities(plant_context, gripper_index_);
}

void PlanarGripper::SetGripperVelocity(
    const drake::systems::Context<double>& diagram_context,
    systems::State<double>* state,
    const Eigen::Ref<const drake::VectorX<double>>& v) const {
  const int num_gripper_velocities = plant_->num_velocities(gripper_index_);
  DRAKE_DEMAND(state != nullptr);
  DRAKE_DEMAND(v.size() == num_gripper_velocities);
  auto& plant_context = this->GetSubsystemContext(*plant_, diagram_context);
  auto& plant_state = this->GetMutableSubsystemState(*plant_, state);
  plant_->SetVelocities(plant_context, &plant_state, gripper_index_, v);
}

PlanarGripper::PlanarGripper(double time_step, bool use_position_control)
    : owned_plant_(std::make_unique<MultibodyPlant<double>>(time_step)),
      owned_scene_graph_(std::make_unique<SceneGraph<double>>()),
      owned_control_plant_(std::make_unique<MultibodyPlant<double>>(time_step)),
      use_position_control_(use_position_control),
      X_WG_(math::RigidTransformd::Identity()) {
  // This class holds the unique_ptrs explicitly for plant and scene_graph
  // until Finalize() is called (when they are moved into the Diagram). Grab
  // the raw pointers, which should stay valid for the lifetime of the Diagram.
  plant_ = owned_plant_.get();
  control_plant_ = owned_control_plant_.get();
  scene_graph_ = owned_scene_graph_.get();
  plant_->RegisterAsSourceForSceneGraph(scene_graph_);
  scene_graph_->set_name("scene_graph");
  plant_->set_name("plant");

  this->set_name("planar_gripper_diagram");
}

void PlanarGripper::SetupPlanarBrick(std::string orientation) {
  SetupPlant(orientation, "drake/examples/planar_gripper/planar_brick.sdf");
}

void PlanarGripper::SetupPinBrick(std::string orientation) {
  SetupPlant(orientation, "drake/examples/planar_gripper/1dof_brick.sdf");
}

void PlanarGripper::SetupPlant(std::string orientation,
                               std::string brick_file_name) {
  Vector3d gravity = Vector3d::Zero();

  // Make and add the planar_gripper model.
  const std::string full_name =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_gripper.sdf");

  gripper_index_ = Parser(plant_, scene_graph_).AddModelFromFile(full_name);
  WeldGripperFrames<double>(plant_, X_WG_);

  // Adds the brick to be manipulated.
  const std::string brick_full_file_name = FindResourceOrThrow(brick_file_name);
  brick_index_ = Parser(plant_).AddModelFromFile(brick_full_file_name, "brick");

  // When the planar-gripper is welded via WeldGripperFrames(), motion always
  // lies in the world Y-Z plane (because the planar-gripper frame is aligned
  // with the world frame). Therefore, gravity can either point along the world
  // -Z axis (vertical case), or world -X axis (horizontal case).
  if (orientation == "vertical") {
    const multibody::Frame<double>& brick_base_frame =
        plant_->GetFrameByName("brick_base_link", brick_index_);
    plant_->WeldFrames(plant_->world_frame(), brick_base_frame);
    gravity = Vector3d(
        0, 0, -multibody::UniformGravityFieldElement<double>::kDefaultStrength);
  } else if (orientation == "horizontal") {
    plant_->AddJoint<PrismaticJoint>("brick_translate_x_joint",
                                     plant_->world_body(), std::nullopt,
                                     plant_->GetBodyByName("brick_base_link"),
                                     std::nullopt, Vector3d::UnitX());
    gravity = Vector3d(
        -multibody::UniformGravityFieldElement<double>::kDefaultStrength, 0, 0);
  } else {
    throw std::logic_error("Unrecognized 'orientation' flag.");
  }

  // Create the controlled plant. Contains only the fingers (no bricks).
  Parser(control_plant_).AddModelFromFile(full_name);
  WeldGripperFrames<double>(control_plant_, X_WG_);

  // Adds a thin floor that can provide friction against the brick.
  AddFloor(plant_, *scene_graph_);

  // Finalize the simulation and control plants.
  plant_->Finalize();
  control_plant_->Finalize();

  is_plant_finalized_ = true;

  // Set the gravity field.
  plant_->mutable_gravity_field().set_gravity_vector(gravity);
  control_plant_->mutable_gravity_field().set_gravity_vector(gravity);

  const auto& inspector = scene_graph_->model_inspector();
  for (const auto& finger :
       {Finger::kFinger1, Finger::kFinger2, Finger::kFinger3}) {
    fingertip_sphere_geometry_ids_.emplace(
        finger,
        inspector.GetGeometryIdByName(
            plant_->GetBodyFrameIdOrThrow(
                plant_->GetBodyByName(to_string(finger) + "_tip_link").index()),
            geometry::Role::kProximity,
            "planar_gripper::tip_sphere_collision"));
  }
  brick_geometry_id_ = inspector.GetGeometryIdByName(
      plant_->GetBodyFrameIdOrThrow(
          plant_->GetBodyByName("brick_link").index()),
      geometry::Role::kProximity, "brick::box_collision");
}

void PlanarGripper::Finalize() {
  systems::DiagramBuilder<double> builder;
  builder.AddSystem(std::move(owned_plant_));
  builder.AddSystem(std::move(owned_scene_graph_));

  if (use_position_control_) {
    // Create the gains for the inverse dynamics controller. These gains were
    // chosen arbitrarily.
    Vector<double, kNumGripperJoints> Kp, Kd, Ki;
    Kp.setConstant(1500);
    Kd.setConstant(500);
    Ki.setConstant(500);

    auto id_controller =
        builder.AddSystem<systems::controllers::InverseDynamicsController>(
            *control_plant_, Kp, Ki, Kd, false);

    // Connect the ID controller.
    builder.Connect(plant_->get_state_output_port(gripper_index_),
                    id_controller->get_input_port_estimated_state());

    builder.ExportInput(id_controller->get_input_port_desired_state(),
                        "desired_gripper_state");

    // The inverse dynamics controller internally uses a "controlled plant",
    // which contains the gripper model *only* (i.e., no brick). Therefore, its
    // output must be re-mapped to the actuation input of the full "simulation
    // plant", which contains both gripper and brick. The system
    // GeneralizedForceToActuationOrdering fills this role.
    auto force_to_actuation =
        builder.AddSystem<GeneralizedForceToActuationOrdering>(*control_plant_);
    builder.Connect(*id_controller, *force_to_actuation);
    builder.Connect(force_to_actuation->get_output_port(0),
                    plant_->get_actuation_input_port(gripper_index_));
  } else {  // Use torque control.
    builder.ExportInput(plant_->get_actuation_input_port(), "actuation");
  }

  builder.ExportOutput(plant_->get_state_output_port(), "plant_state");
  builder.ExportOutput(plant_->get_state_output_port(gripper_index_),
                       "gripper_state");
  builder.ExportOutput(plant_->get_state_output_port(brick_index_),
                       "brick_state");
  builder.ExportInput(plant_->get_applied_spatial_force_input_port(),
                      "spatial_force");

  // Connect MBP and SG.
  builder.Connect(
      plant_->get_geometry_poses_output_port(),
      scene_graph_->get_source_pose_port(plant_->get_source_id().value()));
  builder.Connect(scene_graph_->get_query_output_port(),
                  plant_->get_geometry_query_input_port());

  // Connect the force sensor evaluator and export the output.
  auto force_sensor_evaluator =
      builder.AddSystem<ForceSensorEvaluator>(*plant_);
  builder.Connect(plant_->get_reaction_forces_output_port(),
                  force_sensor_evaluator->get_input_port(0));

  builder.ExportOutput(force_sensor_evaluator->get_output_port(0),
                       "force_sensor");
  builder.ExportOutput(scene_graph_->get_pose_bundle_output_port(),
                       "pose_bundle");
  builder.ExportOutput(plant_->get_contact_results_output_port(),
                       "contact_results");
  builder.ExportOutput(plant_->get_geometry_poses_output_port(),
                       "geometry_poses");
  // TODO(rcory) Remove this after controller uses force_sensor output, instead
  //  of ForceDemuxer.
  builder.ExportOutput(plant_->get_reaction_forces_output_port(),
                       "reaction_forces");
  builder.ExportOutput(scene_graph_->get_query_output_port(),
                       "scene_graph_query");

  builder.BuildInto(this);
  is_diagram_finalized_ = true;
}

void PlanarGripper::SetGripperPosition(
    const drake::systems::Context<double>& diagram_context,
    systems::State<double>* state,
    const Eigen::Ref<const drake::VectorX<double>>& q) const {
  const int kNumGripperPositions = get_num_gripper_positions();
  DRAKE_DEMAND(state != nullptr);
  DRAKE_DEMAND(q.size() == kNumGripperPositions);
  auto& plant_context = this->GetSubsystemContext(*plant_, diagram_context);
  auto& plant_state = this->GetMutableSubsystemState(*plant_, state);
  plant_->SetPositions(plant_context, &plant_state, gripper_index_, q);
}

VectorX<double> PlanarGripper::MakeBrickPositionVector(
    const std::map<std::string, double>& map_in) {
  const int kNumBrickPositions = get_num_brick_positions();
  if (static_cast<int>(map_in.size()) != kNumBrickPositions) {
    throw std::runtime_error(
        "The number of initial condition positions must match the number of "
        "planar-gripper positions");
  }
  return MakePositionVector(map_in, kNumBrickPositions);
}

VectorX<double> PlanarGripper::MakePositionVector(
    const std::map<std::string, double>& map_in,
    const int num_positions) const {
  VectorX<double> position_vector = VectorX<double>::Zero(num_positions);

  // TODO(rcory) use this code block once MBP supports getting position_start
  //  index for a model instance position vector (not the full plant's position
  //  vector).
  //
  // for (auto & iter : map_in) {
  //   auto joint_pos_start_index =
  //       plant_->GetJointByName(iter.first).position_start(model_instance);
  //   position_vector(joint_pos_start_index) = iter.second;
  // }

  std::map<int, std::string> index_joint_map;
  for (auto& iter : map_in) {
    auto joint_pos_start_index =
        plant_->GetJointByName(iter.first).position_start();
    index_joint_map[joint_pos_start_index] = iter.first;
  }

  // Assume the index_joint_map is ordered according to joint position index,
  // and assume that MBP's SetPositions(model_instance) takes in a position
  // subvector in that ordering.
  int vector_index = 0;
  for (auto& iter : index_joint_map) {
    position_vector(vector_index++) = map_in.at(iter.second);
  }
  return position_vector;
}

VectorX<double> PlanarGripper::MakeBrickVelocityVector(
    const std::map<std::string, double>& map_in) {
  const int kNumBrickVelocities = get_num_brick_velocities();
  if (static_cast<int>(map_in.size()) != kNumBrickVelocities) {
    throw std::runtime_error(
        "The number of initial condition velocities must match the number of "
        "planar-gripper velocities");
  }
  return MakeVelocityVector(map_in, kNumBrickVelocities);
}

VectorX<double> PlanarGripper::MakeVelocityVector(
    const std::map<std::string, double>& map_in,
    const int num_velocities) const {
  VectorX<double> velocity_vector = VectorX<double>::Zero(num_velocities);

  // TODO(rcory) use this code block once MBP supports getting velocity_start
  //  index for a model instance velocity vector (not the full plant's velocity
  //  vector).
  //
  // for (auto & iter : map_in) {
  //   auto joint_vel_start_index =
  //       plant_->GetJointByName(iter.first).velocity_start(model_instance);
  //   velocity_vector(joint_vel_start_index) = iter.second;
  // }

  std::map<int, std::string> index_joint_map;
  for (auto& iter : map_in) {
    auto joint_vel_start_index =
        plant_->GetJointByName(iter.first).velocity_start();
    index_joint_map[joint_vel_start_index] = iter.first;
  }

  // Assume the index_joint_map is ordered according to joint velocity index,
  // and assume that MBP's SetVelocities(model_instance) takes in a velocity
  // subvector in that ordering.
  int vector_index = 0;
  for (auto& iter : index_joint_map) {
    velocity_vector(vector_index++) = map_in.at(iter.second);
  }
  return velocity_vector;
}

void PlanarGripper::SetBrickPosition(
    const drake::systems::Context<double>& diagram_context,
    drake::systems::State<double>* state,
    const Eigen::Ref<const VectorX<double>>& q) const {
  const int kNumBrickPositions = get_num_brick_positions();
  DRAKE_DEMAND(state != nullptr);
  DRAKE_DEMAND(q.size() == kNumBrickPositions);
  auto& plant_context = this->GetSubsystemContext(*plant_, diagram_context);
  auto& plant_state = this->GetMutableSubsystemState(*plant_, state);
  plant_->SetPositions(plant_context, &plant_state, brick_index_, q);
}

void PlanarGripper::SetBrickVelocity(
    const drake::systems::Context<double>& diagram_context,
    systems::State<double>* state,
    const Eigen::Ref<const drake::VectorX<double>>& v) const {
  const int num_brick_velocities = plant_->num_velocities(brick_index_);
  DRAKE_DEMAND(state != nullptr);
  DRAKE_DEMAND(v.size() == num_brick_velocities);
  auto& plant_context = this->GetSubsystemContext(*plant_, diagram_context);
  auto& plant_state = this->GetMutableSubsystemState(*plant_, state);
  plant_->SetVelocities(plant_context, &plant_state, brick_index_, v);
}

double PlanarGripper::GetBrickPinJointDamping() const {
  std::string joint_name = "brick_revolute_x_joint";
  if (!plant_->HasJointNamed(joint_name)) {
    throw std::logic_error("Joint " + joint_name +
                           " does not exist in the MBP");
  }
  return plant_->GetJointByName<RevoluteJoint>(joint_name).damping();
}

Vector3d PlanarGripper::GetBrickMoments() const {
  std::string frame_name = "brick_link";
  if (!plant_->HasFrameNamed(frame_name)) {
    throw std::logic_error("Frame " + frame_name +
                           " does not exist in the MBP");
  }
  return dynamic_cast<const multibody::RigidBody<double>&>(
             plant_->GetFrameByName(frame_name).body())
      .default_rotational_inertia()
      .get_moments();
}

double PlanarGripper::GetBrickMass() const {
  std::string frame_name = "brick_link";
  if (!plant_->HasFrameNamed(frame_name)) {
    throw std::logic_error("Frame " + frame_name +
                           " does not exist in the MBP");
  }
  return dynamic_cast<const multibody::RigidBody<double>&>(
             plant_->GetFrameByName("brick_link").body())
      .default_mass();
}

std::pair<std::unordered_set<BrickFace>, Eigen::Vector3d>
PlanarGripper::GetClosestFacesToFinger(
    const systems::Context<double>& plant_context, Finger finger) const {
  const auto& inspector = scene_graph_->model_inspector();

  const auto finger_sphere_geometry_id =
      fingertip_sphere_geometry_ids_.at(finger);

  const auto& query_port = plant_->get_geometry_query_input_port();
  const auto& query_object =
      query_port.template Eval<geometry::QueryObject<double>>(plant_context);
  const geometry::SignedDistancePair<double> signed_distance_pair =
      query_object.ComputeSignedDistancePairClosestPoints(
          finger_sphere_geometry_id, brick_geometry_id_);
  const geometry::Box& box_shape = dynamic_cast<const geometry::Box&>(
      inspector.GetShape(brick_geometry_id_));
  std::unordered_set<BrickFace> closest_faces;
  const Eigen::Vector3d p_BCb =
      inspector.GetPoseInFrame(brick_geometry_id_) * signed_distance_pair.p_BCb;
  if (std::abs(signed_distance_pair.p_BCb(1) - box_shape.depth() / 2) < 1e-3) {
    closest_faces.insert(BrickFace::kPosY);
  } else if (std::abs(signed_distance_pair.p_BCb(1) + box_shape.depth() / 2) <
             1e-3) {
    closest_faces.insert(BrickFace::kNegY);
  }
  if (std::abs(signed_distance_pair.p_BCb(2) - box_shape.height() / 2) < 1e-3) {
    closest_faces.insert(BrickFace::kPosZ);
  } else if (std::abs(signed_distance_pair.p_BCb(2) + box_shape.height() / 2) <
             1e-3) {
    closest_faces.insert(BrickFace::kNegZ);
  }
  return std::make_pair(closest_faces, p_BCb);
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
