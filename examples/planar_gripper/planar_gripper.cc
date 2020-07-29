#include "drake/examples/planar_gripper/planar_gripper.h"

#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/examples/planar_gripper/planar_gripper_lcm.h"
#include "drake/examples/planar_gripper/planar_gripper_utils.h"
#include "drake/geometry/render/render_engine_vtk_factory.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/demultiplexer.h"
#include "drake/systems/sensors/image_to_lcm_image_array_t.h"
#include "drake/systems/sensors/pixel_types.h"

namespace drake {
namespace examples {
namespace planar_gripper {

using Eigen::Vector3d;
using geometry::SceneGraph;
using geometry::render::MakeRenderEngineVtk;
using geometry::render::RenderEngineVtkParams;
using math::RigidTransformd;
using math::RotationMatrix;
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

  const double kFloorHeight = 0.05;
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

void PlanarGripper::SetInverseDynamicsControlGains(
    const Eigen::Ref<const VectorX<double>>& Kp,
    const Eigen::Ref<const VectorX<double>>& Ki,
    const Eigen::Ref<const VectorX<double>>& Kd) {
  if (Kp.size() != kNumGripperJoints || Ki.rows() != kNumGripperJoints ||
      Kd.rows() != kNumGripperJoints) {
    throw std::logic_error(
        "SetInverseDynamicsCOntrolGains: Incorrect vector sizes.");
  }
  Kp_ = Kp; Ki_ = Ki; Kd_ = Kd;
}

void PlanarGripper::GetInverseDynamicsControlGains(
    EigenPtr<VectorX<double>> Kp,
    EigenPtr<VectorX<double>> Ki,
    EigenPtr<VectorX<double>> Kd) {
  if (Kp->rows() != kNumGripperJoints || Ki->rows() != kNumGripperJoints ||
      Kd->rows() != kNumGripperJoints) {
    throw std::logic_error(
        "GetInverseDynamicsControlGains: Incorrect vector sizes.");
  }
  *Kp = Kp_; *Ki = Ki_; *Kd = Kd_;
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
            Value<std::vector<multibody::SpatialForce<double>>>());
    this->DeclareVectorOutputPort("force_sensors_out",
                                  systems::BasicVector<double>(num_sensors * 2),
                                  &ForceSensorEvaluator::SetOutput);
  }

  void SetOutput(const drake::systems::Context<double>& context,
                 drake::systems::BasicVector<double>* output) const {
    const std::vector<multibody::SpatialForce<double>>& spatial_vec =
        this->GetInputPort("spatial_forces_in")
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

/**
 * If the PlanarGripper control type is ControlType::kHybrid, this system
 * allows switching controllers on the fly (between joint torque control and joint
 * position control) for a single finger. The constructor takes in the
 * number of joints on this finger. This system declares three input ports. The
 * first two are vector input ports of size kNumJointsPerFinger. The first of
 * these takes in the position control actuation command (typically the output
 * of the InverseDynamicsController), and the second takes in the joint torque
 * command (typically the output of the ForceController). The third input port
 * is an abstract input port of type `ControlType`, which indicates which
 * actuation command (position-control or torque-control) should be copied to
 * the output. The vector valued inputs/outputs are ordered according to a
 * preferred finger joint ordering, as given by
 * GetPreferredFingerJointOrdering().
 */
class HybridControlSwitch : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(HybridControlSwitch);
  explicit HybridControlSwitch(Finger finger)
      : finger_(finger) {
    // Input ports.
    this->DeclareVectorInputPort(
        "position_control_u",
        systems::BasicVector<double>(kNumJointsPerFinger));
    this->DeclareVectorInputPort(
        "torque_control_u",
        systems::BasicVector<double>(kNumJointsPerFinger));
    this->DeclareAbstractInputPort("control_type", Value<ControlType>());

    // Output ports.
    this->DeclareVectorOutputPort(
        "u_output", systems::BasicVector<double>(kNumJointsPerFinger),
        &HybridControlSwitch::SetOutput);
  }

  Finger get_finger() const {
    return finger_;
  }

 private:
  void SetOutput(const drake::systems::Context<double>& context,
                 drake::systems::BasicVector<double>* u_output) const {
    const ControlType& control_type =
        this->GetInputPort("control_type").Eval<ControlType>(context);
    if (control_type == ControlType::kPosition) {
      auto position_port_index =
          this->GetInputPort("position_control_u").get_index();
      VectorX<double> u_position_input =
          this->EvalVectorInput(context, position_port_index)->get_value();
      DRAKE_DEMAND(u_position_input.size() == kNumJointsPerFinger);
      u_output->get_mutable_value() = u_position_input;
    } else if (control_type == ControlType::kTorque) {
      auto torque_port_index =
          this->GetInputPort("torque_control_u").get_index();
      VectorX<double> u_torque_input =
          this->EvalVectorInput(context, torque_port_index)->get_value();
      DRAKE_DEMAND(u_torque_input.size() == kNumJointsPerFinger);
      u_output->get_mutable_value() = u_torque_input;
    } else {
      throw std::logic_error(
          "HybridControlSwitcher: Control type input port options are "
          "{kPosition, "
          "kTorque}");
    }
  }

  // Indicates which finger this switcher is associated with.
  const Finger finger_;
};

/**
 * This system declares a set of n input ports each of size (2 *
 * kNumJointsPerFinger), where n is the number of fingers in the plant. Input
 * port n corresponds to finger n, and takes in the state of finger n in the
 * order: {fn_positions, fn_velocities}, where positions and velocities
 * are ordered according to GetPreferredFingerJointOrdering(). The system
 * declares a single output port of size (2 * kNumGripperJoints), which contains
 * the state of the planar gripper in MBP joint velocity index ordering (e.g.,
 * as needed by the desired state input port of the InverseDynamicsController).
 */
class FingersStateToGripperState final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FingersStateToGripperState);
  explicit FingersStateToGripperState(const PlanarGripper& planar_gripper)
      : planar_gripper_(planar_gripper) {
    // Define separate desired state inputs for each of the fingers.
    for (int i = 1; i <= kNumFingers; i++) {
      this->DeclareInputPort(to_string_from_finger_num(i) + "_desired_state",
                             systems::kVectorValued, 2 * kNumJointsPerFinger);
    }
    this->DeclareVectorOutputPort(
        "gripper_desired_state",
        systems::BasicVector<double>(kNumGripperJoints * 2),
        &FingersStateToGripperState::SetOutput);
  }

 private:
  void SetOutput(const systems::Context<double>& context,
                 systems::BasicVector<double>* output_vector) const {
    // This defines the joint name to value pairs for the concatenated
    // finger positions/velocities extracted from the input ports.
    std::map<std::string, double> gripper_position_map;
    std::map<std::string, double> gripper_velocity_map;

    // Get the preferred finger joint ordering.
    std::vector<std::string> finger_joint_ordering =
        GetPreferredFingerJointOrdering();

    // Record position and velocity values ordered by finger number.
    for (int i = 1; i <= kNumFingers; i++) {
      auto finger_name = to_string_from_finger_num(i);
      auto finger_port_index =
          this->GetInputPort(finger_name + "_desired_state").get_index();
      auto finger_state =
          this->EvalVectorInput(context, finger_port_index)->get_value();
      DRAKE_DEMAND(finger_state.size() == (2 * kNumJointsPerFinger));

      int joint_index = 0;
      for (const auto& joint_name : finger_joint_ordering) {
        std::string full_joint_name = finger_name + "_" + joint_name;
        gripper_position_map[full_joint_name] =
            finger_state(joint_index);
        gripper_velocity_map[full_joint_name] =
            finger_state(kNumJointsPerFinger + joint_index);
        joint_index++;
      }
    }
    VectorX<double> gripper_state(2 * kNumGripperJoints);
    gripper_state.head<kNumGripperJoints>() =
        planar_gripper_.MakeGripperPositionVector(gripper_position_map);
    gripper_state.tail<kNumGripperJoints>() =
        planar_gripper_.MakeGripperVelocityVector(gripper_velocity_map);
    output_vector->get_mutable_value() = gripper_state;
  }
  // TODO(rcory) Eliminate the need for this member variable once
  //  MakeGripperPositionVector and MakeGripperVelocityVector methods are moved
  //  to common utilities.
  const PlanarGripper& planar_gripper_;
};

/**
 * This system takes an input vector u of actuator values for the planar gripper
 * (no brick), with size kNumGripperJoints. The input vector is assumed to be
 * ordered according to joint actuator index, ensuring we can connect the output
 * of GeneralizedForceToActuationOrdering to this system's input. This system
 * declares a single output port which produces a reordered actuation vector
 * according to GetPreferredGripperJointOrdering().
 */
class GripperActuationToPreferredOrdering final
    : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GripperActuationToPreferredOrdering);
  explicit GripperActuationToPreferredOrdering(
      const MultibodyPlant<double>& plant) {
    this->DeclareVectorInputPort(
        "gripper_u_in", systems::BasicVector<double>(kNumGripperJoints));
    this->DeclareVectorOutputPort(
        "gripper_u_out", systems::BasicVector<double>(kNumGripperJoints),
        &GripperActuationToPreferredOrdering::SetOutput);
    std::vector<std::string> preferred_joint_ordering =
        GetPreferredGripperJointOrdering();
    std::vector<multibody::JointIndex> joint_index_vector;
    for (const auto& iter : preferred_joint_ordering) {
      joint_index_vector.push_back(plant.GetJointByName(iter).index());
    }
    // Create the Sᵤ matrix inverse.
    actuation_selector_matrix_inv_ =
        plant.MakeActuatorSelectorMatrix(joint_index_vector).inverse();
  }

 private:
  void SetOutput(const systems::Context<double>& context,
                 systems::BasicVector<double>* output_vector) const {
    const auto input_port_index =
        this->GetInputPort("gripper_u_in").get_index();
    VectorX<double> gripper_u =
        this->EvalVectorInput(context, input_port_index)->get_value();
    DRAKE_DEMAND(gripper_u.size() == kNumGripperJoints);

    // Reorder the gripper actuation input vector to the preferred ordering.
    VectorX<double> gripper_u_reordered =
        actuation_selector_matrix_inv_ * gripper_u;

    DRAKE_DEMAND(gripper_u_reordered.size() == kNumGripperJoints);
    output_vector->get_mutable_value() = gripper_u_reordered;
  }

  MatrixX<double> actuation_selector_matrix_inv_;
};

/**
 * This system declares kNumFingers input ports, taking in vectors of size
 * kNumJointsPerFinger, where each input vector consists of actuator values for
 * a single finger (ordered by GetPreferredFingerJointOrdering()). The system
 * declares a single output port, which outputs a vector u of planar gripper
 * actuation values, ordered according to the plant's joint actuator index, and
 * is of size kNumGripperJoints.
 */
class FingersToGripperActuation final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FingersToGripperActuation);
  explicit FingersToGripperActuation(const MultibodyPlant<double>& plant) {
    // Define separate actuation inputs for each of the fingers.
    for (int i = 1; i <= kNumFingers; i++) {
      std::string finger_name = to_string_from_finger_num(i);
      this->DeclareInputPort(finger_name + "_u",
                             systems::kVectorValued, kNumJointsPerFinger);
    }
    this->DeclareVectorOutputPort(
        "gripper_u", systems::BasicVector<double>(kNumGripperJoints),
        &FingersToGripperActuation::SetOutput);

    std::vector<std::string> preferred_joint_ordering =
        GetPreferredGripperJointOrdering();
    std::vector<multibody::JointIndex> joint_index_vector;
    for (const auto& iter : preferred_joint_ordering) {
      joint_index_vector.push_back(plant.GetJointByName(iter).index());
    }
    // Create the Sᵤ matrix.
    actuation_selector_matrix_ =
        plant.MakeActuatorSelectorMatrix(joint_index_vector);
  }

 private:
  void SetOutput(const systems::Context<double>& context,
                 systems::BasicVector<double>* output_vector) const {
    VectorX<double> gripper_u(kNumGripperJoints);  /* in preferred ordering */

    // Record actuation values ordered by finger number.
    for (int i = 0; i < kNumFingers; i++) {
      auto finger_name = to_string_from_finger_num(i + 1);
      auto finger_port_index =
          this->GetInputPort(finger_name + "_u").get_index();
      auto finger_u =
          this->EvalVectorInput(context, finger_port_index)->get_value();
      DRAKE_DEMAND(finger_u.size() == kNumJointsPerFinger);
      gripper_u.segment(i * 2, kNumJointsPerFinger) = finger_u;
    }
    // Convert to plant's actuator index ordering.
    output_vector->get_mutable_value() = actuation_selector_matrix_ * gripper_u;
  }

  // Selector matrix that converts an actuation vector in preferred ordering to
  // an actuation vector in MBP actuator index ordering.
  MatrixX<double> actuation_selector_matrix_;
};

VectorX<double> PlanarGripper::MakeGripperPositionVector(
    const std::map<std::string, double>& map_in) const {
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

VectorX<double> PlanarGripper::MakeGripperVelocityVector(
    const std::map<std::string, double>& map_in) const {
  const int kNumGripperVelocities = get_num_gripper_velocities();
  if (static_cast<int>(map_in.size()) != kNumGripperVelocities) {
    throw std::runtime_error(
        "The number of initial condition velocities must match the number of "
        "planar-gripper velocities");
  }
  return MakeVelocityVector(map_in, kNumGripperVelocities);
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
  DRAKE_DEMAND(state != nullptr);
  DRAKE_DEMAND(v.size() == get_num_gripper_velocities());
  auto& plant_context = this->GetSubsystemContext(*plant_, diagram_context);
  auto& plant_state = this->GetMutableSubsystemState(*plant_, state);
  plant_->SetVelocities(plant_context, &plant_state, gripper_index_, v);
}

PlanarGripper::PlanarGripper(double time_step, ControlType control_type,
                             bool add_floor)
    : owned_control_plant_(std::make_unique<MultibodyPlant<double>>(time_step)),
      owned_plant_(std::make_unique<MultibodyPlant<double>>(time_step)),
      owned_scene_graph_(std::make_unique<SceneGraph<double>>()),
      control_type_(control_type),
      add_floor_(add_floor),
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

  // Add a default renderer, which will be needed if a camera is added.
  scene_graph_->AddRenderer(default_renderer_name_,
                            MakeRenderEngineVtk(RenderEngineVtkParams()));

  // Create the default gains for the inverse dynamics controller. These gains
  // were chosen arbitrarily.
  Kp_.setConstant(1500);
  Kd_.setConstant(500);
  Ki_.setConstant(500);

  this->set_name("planar_gripper_diagram");
}

void PlanarGripper::SetupPlanarBrick(std::string orientation) {
  SetupPlant(orientation,
             "drake/examples/planar_gripper/models/planar_brick.sdf");
}

void PlanarGripper::SetupPinBrick(std::string orientation) {
  SetupPlant(orientation,
             "drake/examples/planar_gripper/models/1dof_brick.sdf");
}

void PlanarGripper::SetupPlant(std::string orientation,
                               std::string brick_file_name) {
  Vector3d gravity = Vector3d::Zero();

  // Make and add the planar_gripper model.
  const std::string gripper_full_name = FindResourceOrThrow(
      "drake/examples/planar_gripper/models/planar_gripper.sdf");

  gripper_index_ = Parser(plant_, scene_graph_)
                       .AddModelFromFile(gripper_full_name, "planar_gripper");
  WeldGripperFrames<double>(plant_, X_WG_);

  // Adds the brick to be manipulated.
  const std::string brick_full_file_name = FindResourceOrThrow(brick_file_name);
  brick_index_ = Parser(plant_, scene_graph_)
                     .AddModelFromFile(brick_full_file_name, "brick");

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
  Parser(control_plant_).AddModelFromFile(gripper_full_name);
  WeldGripperFrames<double>(control_plant_, X_WG_);

  if (add_floor_) {
    // Adds a thin floor that can provide friction against the brick.
    AddFloor(plant_, *scene_graph_);
  }

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
        finger, GetFingertipSphereGeometryId(*plant_, inspector, finger));
  }
  brick_geometry_id_ = GetBrickGeometryId(*plant_, inspector);
}

systems::sensors::RgbdSensor* PlanarGripper::AddCamera(
    const systems::OutputPort<double>& query_output_port,
    const std::string& camera_name, systems::DiagramBuilder<double>* builder) {
  DRAKE_DEMAND(builder != nullptr);

  // Properties of a simple pin-hole camera.
  // TODO(huihua) Move these parameters to a configuration file.
  constexpr int kWidth = 1080;    // [pixel]
  constexpr int kHeight = 720;    // [pixel]
  constexpr double kFovY = 1.0;   // [rad]
  constexpr double kZNear = 0.1;  // [m]
  constexpr double kZFar = 5;     // [m]
  geometry::render::CameraProperties color_properties(kWidth, kHeight, kFovY,
                                                      default_renderer_name_);
  geometry::render::DepthCameraProperties depth_properties(
      kWidth, kHeight, kFovY, default_renderer_name_, kZNear, kZFar);

  // Confirm the mount frame is valid. The camera is mounted on the brick base
  // link, which is welded to the ground.
  const auto& mount_frame = plant_->GetFrameByName("brick_base_link");
  const std::optional<geometry::FrameId> mount_frame_id =
      plant_->GetBodyFrameIdIfExists(mount_frame.body().index());
  DRAKE_THROW_UNLESS(mount_frame_id.has_value());

  // Set the position and orintation of the camera w.r.t to the brick base link.
  // The Z rotation is to make sure the image has the same orientation as the
  // drake visualizer. This will yield a more intuitive visual feedback from the
  // image.
  const RotationMatrix<double> R_BC =
      RotationMatrix<double>::MakeYRotation(-M_PI_2) *
      RotationMatrix<double>::MakeZRotation(M_PI_2);
  constexpr double kCameraZOffset = 0.4;  // [m]
  drake::Vector3<double> p_BC_B{kCameraZOffset, 0.0, 0.0};
  const RigidTransformd X_BC(R_BC, p_BC_B);
  const RigidTransformd X_PC = mount_frame.GetFixedPoseInBodyFrame() * X_BC;

  auto camera = builder->template AddSystem<systems::sensors::RgbdSensor>(
      mount_frame_id.value(), X_PC, color_properties, depth_properties);
  camera->set_name(camera_name);

  builder->Connect(query_output_port, camera->query_object_input_port());

  // Publish the camera image (both rgb and depth) for visualization.
  auto image_to_lcm_image_array =
      builder->AddSystem<drake::systems::sensors::ImageToLcmImageArrayT>();
  image_to_lcm_image_array->set_name(camera_name + "_viewer");

  const auto& color_cam_port =
      image_to_lcm_image_array
          ->DeclareImageInputPort<drake::systems::sensors::PixelType::kRgba8U>(
              "color_camera_" + camera_name);
  builder->Connect(camera->color_image_output_port(), color_cam_port);

  const auto& depth_cam_port = image_to_lcm_image_array->DeclareImageInputPort<
      drake::systems::sensors::PixelType::kDepth16U>("depth_camera_" +
                                                     camera_name);
  builder->Connect(camera->depth_image_16U_output_port(), depth_cam_port);

  builder->ExportOutput(
      image_to_lcm_image_array->image_array_t_msg_output_port(),
      camera_name + "_images");
  return camera;
}

// Adds an inverse dynamics controller and internally remaps its generalized
// forces to actuation vector u, which is then connected to `u_input` input
// port.
systems::controllers::InverseDynamicsController<double>*
PlanarGripper::AddInverseDynamicsController(
    const systems::InputPort<double>& u_input,
    systems::DiagramBuilder<double>* builder) {
  systems::controllers::InverseDynamicsController<double>* id_controller;
  id_controller =
      builder->AddSystem<systems::controllers::InverseDynamicsController>(
          *control_plant_, Kp_, Ki_, Kd_, false);
  id_controller->set_name("inverse_dynamics_controller");

  // Connect the ID controller.
  builder->Connect(plant_->get_state_output_port(gripper_index_),
                  id_controller->get_input_port_estimated_state());

  // The inverse dynamics controller internally uses a "controlled plant",
  // which contains the gripper model *only* (i.e., no brick). Therefore, its
  // output must be re-mapped to the actuation input of the full "simulation
  // plant", which contains both gripper and brick. The system
  // GeneralizedForceToActuationOrdering fills this role.
  auto force_to_actuation =
      builder->AddSystem<GeneralizedForceToActuationOrdering>(*control_plant_);
  force_to_actuation->set_name("force_to_actuation_ordering");
  builder->Connect(*id_controller, *force_to_actuation);
  builder->Connect(force_to_actuation->GetOutputPort("u"), u_input);

  return id_controller;
}

void PlanarGripper::Finalize() {
  systems::DiagramBuilder<double> builder;
  builder.AddSystem(std::move(owned_plant_))->set_name("multibody_plant");
  builder.AddSystem(std::move(owned_scene_graph_))->set_name("scene_graph");

  if (control_type_ == ControlType::kPosition) {
    auto id_controller = AddInverseDynamicsController(
        plant_->get_actuation_input_port(gripper_index_), &builder);
    builder.ExportInput(id_controller->get_input_port_desired_state(),
                        "desired_gripper_state");
  } else if (control_type_ == ControlType::kTorque) {
    builder.ExportInput(plant_->get_actuation_input_port(), "torque_control_u");
  } else if (control_type_ == ControlType::kHybrid) {
    // System to convert individual finger states to a concatenated gripper
    // state.
    auto fingers_to_gripper_desired_state_sys =
        builder.AddSystem<FingersStateToGripperState>(*this);

    // System that reorders all fingers' u vectors, which are output from the
    // hybrid switcher system and then input to this system. Provides an output
    // with the entire gripper u in the plant's actuator ordering.
    auto fingers_to_gripper_actuation =
        builder.AddSystem<FingersToGripperActuation>(*plant_);

    // System that reorders the actuation vector u for the gripper. Input is in
    // actuator index ordering, and output is in preferred ordering.
    auto gripper_u_reorder_sys =
        builder.AddSystem<GripperActuationToPreferredOrdering>(*plant_);

    // Adds an inverse dynamics controller and internally remaps its generalized
    // forces to actuation vector u, which is then connected to the input port
    // passed in here.
    auto id_controller = AddInverseDynamicsController(
        gripper_u_reorder_sys->GetInputPort("gripper_u_in"), &builder);

    builder.Connect(fingers_to_gripper_desired_state_sys->GetOutputPort(
        "gripper_desired_state"),
                    id_controller->get_input_port_desired_state());

    // Demuxer to reorder the gripper u (originating from inverse dynamics
    // control) into individual finger u's. Note that the reordered input into
    // the demuxer is already ordered according to a preferred ordering.
    auto idc_u_demuxer = builder.AddSystem<systems::Demultiplexer<double>>(
        kNumGripperJoints, kNumJointsPerFinger);
    builder.Connect(gripper_u_reorder_sys->GetOutputPort("gripper_u_out"),
                    idc_u_demuxer->get_input_port(0));

    // Now connect the concatenated actuation vector u (ordered in the plant's
    // actuator ordering) to the plant's actuation input port.
    builder.Connect(fingers_to_gripper_actuation->GetOutputPort("gripper_u"),
                    plant_->get_actuation_input_port());

    for (int i = 1; i <= kNumFingers; i++) {  /* iterate over finger numbers */
      std::string finger_name = to_string_from_finger_num(i);

      // Export an input port for each finger's desired state.
      builder.ExportInput(fingers_to_gripper_desired_state_sys->GetInputPort(
          finger_name + "_desired_state"), finger_name + "_desired_state");

      // Each finger can individually operate in position or force control, so
      // we create a hybrid switch for each finger in the plant. We also export
      // the torque control input port and control-type input port for each
      // finger.
      auto hybrid_control_switch =
          builder.AddSystem<HybridControlSwitch>(to_Finger(i));
      builder.ExportInput(
          hybrid_control_switch->GetInputPort("torque_control_u"),
          finger_name + "_torque_control_u");
      builder.ExportInput(hybrid_control_switch->GetInputPort("control_type"),
                          finger_name + "_control_type");

      // Connect the output of the demuxer for this finger into the
      // corresponding input of this finger's switcher.
      builder.Connect(
          idc_u_demuxer->get_output_port(i - 1),
          hybrid_control_switch->GetInputPort("position_control_u"));

      // Connect the output of the switcher to the appropriate input port of the
      // finger-to-gripper actuation concatenator.
      builder.Connect(
          hybrid_control_switch->GetOutputPort("u_output"),
          fingers_to_gripper_actuation->GetInputPort(finger_name + "_u"));
    }
  } else {
    throw std::runtime_error(
        "Unknown control type. Options are {kPosition, kTorque, kHybrid}.");
  }

  // Add an rgbd camera that sits on top of the planar gripper platform. The
  // camera points exactly to the center of the platform.
  AddCamera(scene_graph_->get_query_output_port(), default_camera_name_,
            &builder);

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
  DRAKE_DEMAND(state != nullptr);
  DRAKE_DEMAND(q.size() == get_num_gripper_positions());
  auto& plant_context = this->GetSubsystemContext(*plant_, diagram_context);
  auto& plant_state = this->GetMutableSubsystemState(*plant_, state);
  plant_->SetPositions(plant_context, &plant_state, gripper_index_, q);
}

VectorX<double> PlanarGripper::MakeBrickPositionVector(
    const std::map<std::string, double>& map_in) {
  if (static_cast<int>(map_in.size()) != get_num_brick_positions()) {
    throw std::runtime_error(
        "The number of initial condition positions must match the number of "
        "planar-gripper positions");
  }
  return MakePositionVector(map_in, get_num_brick_positions());
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
        "brick velocities");
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

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
