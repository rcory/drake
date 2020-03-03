#include "drake/examples/planar_gripper/planar_gripper_lcm.h"

#include <utility>
#include <vector>

#include "drake/common/drake_assert.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/lcmt_planar_gripper_command.hpp"
#include "drake/lcmt_planar_gripper_status.hpp"
#include "drake/lcmt_planar_plant_state.hpp"
#include "drake/lcmt_planar_manipuland_spatial_forces.hpp"
#include "drake/lcmt_planar_manipuland_desired.hpp"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"

namespace drake {
namespace examples {
namespace planar_gripper {

using systems::BasicVector;
using systems::Context;
using systems::DiscreteValues;
using systems::State;
using systems::DiscreteUpdateEvent;
using systems::UnrestrictedUpdateEvent;

GripperCommandDecoder::GripperCommandDecoder(int num_fingers) :
      num_fingers_(num_fingers), num_joints_(num_fingers * 2) {
  this->DeclareAbstractInputPort(
      "lcmt_planar_gripper_command",
      Value<lcmt_planar_gripper_command>{});
  state_output_port_ = &this->DeclareVectorOutputPort(
      "state", systems::BasicVector<double>(num_joints_ * 2),
      &GripperCommandDecoder::OutputStateCommand);
  torques_output_port_ = &this->DeclareVectorOutputPort(
      "torques", systems::BasicVector<double>(num_joints_),
      &GripperCommandDecoder::OutputTorqueCommand);
  this->DeclarePeriodicDiscreteUpdateEvent(
      kGripperLcmPeriod, 0., &GripperCommandDecoder::UpdateDiscreteState);
  // Register a forced discrete state update event. It is added for unit test,
  // or for potential users who require forced updates.
  this->DeclareForcedDiscreteUpdateEvent(
      &GripperCommandDecoder::UpdateDiscreteState);
  // Discrete state holds pos, vel, torque.
  this->DeclareDiscreteState(num_joints_ * 3);
}

void GripperCommandDecoder::set_initial_position(
    Context<double>* context,
    const Eigen::Ref<const VectorX<double>> pos) const {
  // The Discrete state consists of positions, velocities, torques.
  auto state_value =
      context->get_mutable_discrete_state(0).get_mutable_value();
  DRAKE_ASSERT(pos.size() == num_joints_);
  // Set the initial positions.
  state_value.head(num_joints_) = pos;
  // Set the initial velocities and torques to zero.
  state_value.tail(num_joints_ * 2) = VectorX<double>::Zero(num_joints_ * 2);
}

systems::EventStatus GripperCommandDecoder::UpdateDiscreteState(
    const Context<double>& context,
    DiscreteValues<double>* discrete_state) const {
  const AbstractValue* input = this->EvalAbstractInput(context, 0);
  DRAKE_ASSERT(input != nullptr);
  const auto& command = input->get_value<lcmt_planar_gripper_command>();

  // If we're using a default constructed message (haven't received
  // a command yet), keep using the initial state.
  BasicVector<double>& state = discrete_state->get_mutable_vector(0);
  auto state_value = state.get_mutable_value();
  auto positions = state_value.head(num_joints_);
  auto velocities = state_value.segment(num_joints_, num_joints_);
  auto torques = state_value.tail(num_joints_);

  DRAKE_DEMAND(command.num_fingers == 0 || command.num_fingers == num_fingers_);

  for (int i = 0; i < command.num_fingers; ++i) {
    const lcmt_planar_gripper_finger_command& fcommand =
        command.finger_command[i];
    const int st_index = 2 * i;
    positions(st_index) = fcommand.joint_position[0];
    positions(st_index + 1) = fcommand.joint_position[1];
    velocities(st_index) = fcommand.joint_velocity[0];
    velocities(st_index + 1) = fcommand.joint_velocity[1];
    torques(st_index) = fcommand.joint_torque[0];
    torques(st_index + 1) = fcommand.joint_torque[1];
  }

  return systems::EventStatus::Succeeded();
}

void GripperCommandDecoder::OutputStateCommand(
    const Context<double>& context, BasicVector<double>* output) const {
  Eigen::VectorBlock<VectorX<double>> output_vec = output->get_mutable_value();
  output_vec = context.get_discrete_state(0).get_value().head(num_joints_ * 2);
}

void GripperCommandDecoder::OutputTorqueCommand(
    const Context<double>& context, BasicVector<double>* output) const {
  Eigen::VectorBlock<VectorX<double>> output_vec =
      output->get_mutable_value();
  output_vec = context.get_discrete_state(0).get_value().tail(num_joints_);
}

GripperCommandEncoder::GripperCommandEncoder(int num_fingers) :
      num_fingers_(num_fingers), num_joints_(num_fingers * 2) {
  state_input_port_ =
      &this->DeclareInputPort("state", systems::kVectorValued, num_joints_ * 2);
  torques_input_port_ =
      &this->DeclareInputPort("torque", systems::kVectorValued, num_joints_);
  this->DeclareAbstractOutputPort("lcmt_gripper_command",
                                  &GripperCommandEncoder::OutputCommand);
}

void GripperCommandEncoder::OutputCommand(
    const Context<double>& context,
    lcmt_planar_gripper_command* command) const {
  command->utime = static_cast<int64_t>(context.get_time() * 1e6);
  const systems::BasicVector<double>* state_input =
      this->EvalVectorInput(context, 0);
  const systems::BasicVector<double>* torque_input =
      this->EvalVectorInput(context, 1);

  command->num_fingers = num_fingers_;
  command->finger_command.resize(num_fingers_);

  for (int i = 0; i < num_fingers_; ++i) {
    const int st_index = 2 * i;
    lcmt_planar_gripper_finger_command& fcommand =
        command->finger_command[i];
    fcommand.joint_position[0] = state_input->GetAtIndex(st_index);
    fcommand.joint_position[1] = state_input->GetAtIndex(st_index + 1);
    fcommand.joint_velocity[0] =
        state_input->GetAtIndex(num_joints_ + st_index);
    fcommand.joint_velocity[1] =
        state_input->GetAtIndex(num_joints_ + st_index + 1);
    fcommand.joint_torque[0] = torque_input->GetAtIndex(st_index);
    fcommand.joint_torque[1] = torque_input->GetAtIndex(st_index + 1);
  }
}

GripperStatusDecoder::GripperStatusDecoder(int num_fingers)
    : num_fingers_(num_fingers),
      num_joints_(num_fingers * 2),
      num_tip_forces_(num_fingers * 2) {
  state_output_port_ = &this->DeclareVectorOutputPort(
      "state", systems::BasicVector<double>(num_joints_ * 2),
      &GripperStatusDecoder::OutputStateStatus);
  force_output_port_ = &this->DeclareVectorOutputPort(
      "fingertip_force", systems::BasicVector<double>(num_tip_forces_),
      &GripperStatusDecoder::OutputForceStatus);
  this->DeclareAbstractInputPort("lcmt_planar_gripper_status",
                                 Value<lcmt_planar_gripper_status>{});
  // Discrete state includes: {state, fingertip_force(y,z)}.
  this->DeclareDiscreteState((num_joints_* 2) + (num_fingers_ * 2));

  this->DeclarePeriodicDiscreteUpdateEvent(
      kGripperLcmPeriod, 0., &GripperStatusDecoder::UpdateDiscreteState);
  // Register a forced discrete state update event. It is added for unit test,
  // or for potential users who require forced updates.
  this->DeclareForcedDiscreteUpdateEvent(
      &GripperStatusDecoder::UpdateDiscreteState);
}

systems::EventStatus GripperStatusDecoder::UpdateDiscreteState(
    const Context<double>& context,
    DiscreteValues<double>* discrete_state) const {
  const AbstractValue* input = this->EvalAbstractInput(context, 0);
  DRAKE_ASSERT(input != nullptr);
  const auto& status = input->get_value<lcmt_planar_gripper_status>();

  BasicVector<double>& state = discrete_state->get_mutable_vector(0);

  auto state_value = state.get_mutable_value();
  auto positions = state_value.head(num_joints_);
  auto velocities = state_value.segment(num_joints_, num_joints_);
  auto tip_forces = state_value.tail(num_fingers_ * 2);

  DRAKE_DEMAND(status.num_fingers == 0 || status.num_fingers == num_fingers_);

  for (int i = 0; i < status.num_fingers; ++i) {
    const int st_index = 2 * i;
    const lcmt_planar_gripper_finger_status& fstatus = status.finger_status[i];
    positions(st_index) = fstatus.joint_position[0];
    positions(st_index + 1) = fstatus.joint_position[1];
    velocities(st_index) = fstatus.joint_velocity[0];
    velocities(st_index + 1) = fstatus.joint_velocity[1];
    tip_forces(st_index) = fstatus.fingertip_force.fy;
    tip_forces(st_index + 1) = fstatus.fingertip_force.fz;
  }

  return systems::EventStatus::Succeeded();
}

void GripperStatusDecoder::OutputStateStatus(
    const Context<double>& context, BasicVector<double>* output) const {
  Eigen::VectorBlock<VectorX<double>> output_vec = output->get_mutable_value();
  output_vec = context.get_discrete_state(0).get_value().head(num_joints_ * 2);
}

void GripperStatusDecoder::OutputForceStatus(
    const Context<double>& context, BasicVector<double>* output) const {
  Eigen::VectorBlock<VectorX<double>> output_vec = output->get_mutable_value();
  output_vec = context.get_discrete_state(0).get_value().tail(num_fingers_ * 2);
}

GripperStatusEncoder::GripperStatusEncoder(int num_fingers)
    : num_fingers_(num_fingers),
      num_joints_(num_fingers * 2),
      num_tip_forces_(num_fingers * 2) {
  state_input_port_ =
      &this->DeclareInputPort("state", systems::kVectorValued, num_joints_ * 2);
  force_input_port_ = &this->DeclareInputPort(
      "fingertip_force", systems::kVectorValued, num_tip_forces_);
  this->DeclareAbstractOutputPort("lcmt_gripper_status",
                                  &GripperStatusEncoder::MakeOutputStatus,
                                  &GripperStatusEncoder::OutputStatus);
}

lcmt_planar_gripper_status GripperStatusEncoder::MakeOutputStatus() const {
  lcmt_planar_gripper_status msg{};
  msg.utime = 0;
  msg.num_fingers = num_fingers_;
  msg.finger_status.resize(num_fingers_);
  return msg;
}

void GripperStatusEncoder::OutputStatus(
    const Context<double>& context, lcmt_planar_gripper_status* status) const {
  status->utime = static_cast<int64_t>(context.get_time() * 1e6);
  const systems::BasicVector<double>* state_input =
      this->EvalVectorInput(context, 0);
  auto state_value = state_input->get_value();
  const systems::BasicVector<double>* force_input =
      this->EvalVectorInput(context, 1);
  auto force_value = force_input->get_value();

  status->num_fingers = num_fingers_;
  status->finger_status.resize(num_fingers_);
  for (int i = 0; i < num_fingers_; ++i) {
    const int st_index = 2 * i;
    lcmt_planar_gripper_finger_status& fstatus = status->finger_status[i];
    fstatus.joint_position[0] = state_value(st_index);
    fstatus.joint_position[1] = state_value(st_index + 1);
    fstatus.joint_velocity[0] =
        state_value(num_joints_ + st_index);
    fstatus.joint_velocity[1] =
        state_value(num_joints_ + st_index + 1);
    fstatus.fingertip_force.timestamp =
        static_cast<int64_t>(context.get_time() * 1e3);
    fstatus.fingertip_force.fy = force_value(st_index);
    fstatus.fingertip_force.fz = force_value(st_index + 1);

    // For the planar gripper, these are all zero.
    fstatus.fingertip_force.fx = 0;
    fstatus.fingertip_force.tx = 0;
    fstatus.fingertip_force.ty = 0;
    fstatus.fingertip_force.tz = 0;
  }
}

/// =================== QP Controller Section ===========================
QPBrickControlDecoder::QPBrickControlDecoder(
    multibody::BodyIndex brick_body_index)
    : brick_body_index_(brick_body_index) {
  this->DeclareAbstractInputPort(
      "spatial_forces_lcm",
      Value<lcmt_planar_manipuland_spatial_forces>{});
  this->DeclareAbstractOutputPort("qp_brick_control",
                                  &QPBrickControlDecoder::OutputBrickControl);
  // State holds an abstract value of std::vector of
  // ExternallyAppliedSpatialForce.
  this->DeclareAbstractState(
      std::make_unique<Value<
          std::vector<multibody::ExternallyAppliedSpatialForce<double>>>>());
  this->DeclarePeriodicUnrestrictedUpdateEvent(
      kGripperLcmPeriod, 0.,
      &QPBrickControlDecoder::UpdateAbstractState);
  // Register a forced discrete state update event. It is added for unit test,
  // or for potential users who require forced updates.
  this->DeclareForcedUnrestrictedUpdateEvent(
      &QPBrickControlDecoder::UpdateAbstractState);
}

systems::EventStatus QPBrickControlDecoder::UpdateAbstractState(
    const Context<double>& context, State<double>* state) const {
  lcmt_planar_manipuland_spatial_forces spatial_forces_lcm =
      this->GetInputPort("spatial_forces_lcm")
          .Eval<lcmt_planar_manipuland_spatial_forces>(context);

  // If we've received at least the first lcm message, update the state here.
  if (spatial_forces_lcm.num_forces > 0) {
    auto& brick_control = state->get_mutable_abstract_state<
        std::vector<multibody::ExternallyAppliedSpatialForce<double>>>(0);
    brick_control.clear();
    DRAKE_DEMAND(spatial_forces_lcm.manip_body_name == "brick_link");
    DRAKE_DEMAND(static_cast<int>(spatial_forces_lcm.num_forces) ==
                 static_cast<int>(spatial_forces_lcm.forces.size()));
    for (auto & spatial_force_lcm : spatial_forces_lcm.forces) {
      DRAKE_DEMAND(spatial_force_lcm.manip_body_name == "brick_link");
      multibody::ExternallyAppliedSpatialForce<double> applied_spatial_force;
      applied_spatial_force.body_index = brick_body_index_;
      applied_spatial_force.p_BoBq_B(0) = 0;  // x
      applied_spatial_force.p_BoBq_B(1) = spatial_force_lcm.p_BoBq_B[0];  // y
      applied_spatial_force.p_BoBq_B(2) = spatial_force_lcm.p_BoBq_B[1];  // z

      Eigen::Vector3d force(0, spatial_force_lcm.force_Bq_W[0],
                            spatial_force_lcm.force_Bq_W[1]);

      Eigen::Vector3d torque = Eigen::Vector3d::Zero();
      torque(0) = spatial_force_lcm.torque_Bq_W;  // tau_x

      applied_spatial_force.F_Bq_W =
          multibody::SpatialForce<double>(torque, force);
      brick_control.push_back(applied_spatial_force);
    }
  }
  return systems::EventStatus::Succeeded();
}

void QPBrickControlDecoder::OutputBrickControl(
    const systems::Context<double>& context,
    std::vector<multibody::ExternallyAppliedSpatialForce<double>>*
        brick_control) const {
  brick_control->clear();
  *brick_control = context.get_abstract_state<
      std::vector<multibody::ExternallyAppliedSpatialForce<double>>>(0);
}

QPBrickControlEncoder::QPBrickControlEncoder() {
  this->DeclareAbstractInputPort(
      "qp_brick_control",
      Value<std::vector<multibody::ExternallyAppliedSpatialForce<double>>>{});
  this->DeclareAbstractOutputPort("spatial_forces_lcm",
                                  &QPBrickControlEncoder::EncodeBrickControl);
}

void QPBrickControlEncoder::EncodeBrickControl(
    const drake::systems::Context<double>& context,
    lcmt_planar_manipuland_spatial_forces* spatial_forces_lcm) const {
  std::string brick_body_name = "brick_link";  // hard-coded for now.
  spatial_forces_lcm->utime = static_cast<int64_t>(context.get_time() * 1e6);
  spatial_forces_lcm->manip_body_name = brick_body_name;
  std::vector<multibody::ExternallyAppliedSpatialForce<double>> brick_control =
      this->GetInputPort("qp_brick_control")
          .Eval<std::vector<multibody::ExternallyAppliedSpatialForce<double>>>(
              context);

  spatial_forces_lcm->num_forces = brick_control.size();
  spatial_forces_lcm->forces.clear();
  for (auto& spatial_force : brick_control) {
    lcmt_planar_manipuland_spatial_force spatial_force_lcm;
    spatial_force_lcm.utime = spatial_forces_lcm->utime;
    spatial_force_lcm.manip_body_name = brick_body_name;
    spatial_force_lcm.finger_name = "";
    spatial_force_lcm.p_BoBq_B[0] = spatial_force.p_BoBq_B(1);  // y
    spatial_force_lcm.p_BoBq_B[1] = spatial_force.p_BoBq_B(2);  // z
    spatial_force_lcm.force_Bq_W[0] = spatial_force.F_Bq_W.translational()(1);  // fy
    spatial_force_lcm.force_Bq_W[1] = spatial_force.F_Bq_W.translational()(2);  // fz
    spatial_force_lcm.torque_Bq_W = spatial_force.F_Bq_W.rotational()(0);  // tx.
    spatial_forces_lcm->forces.push_back(spatial_force_lcm);
  }
}

QPFingersControlDecoder::QPFingersControlDecoder(
    multibody::BodyIndex brick_body_index)
    : brick_body_index_(brick_body_index) {
  this->DeclareAbstractInputPort(
      "spatial_forces_lcm",
      Value<lcmt_planar_manipuland_spatial_forces>{});
  this->DeclareAbstractOutputPort(
      "qp_fingers_control", &QPFingersControlDecoder::OutputFingersControl);

  // State holds an abstract value of
  // std::unordered_map<Finger, multibody::ExternallyAppliedSpatialForce<double>>
  this->DeclareAbstractState(
      std::make_unique<Value<std::unordered_map<
          Finger, multibody::ExternallyAppliedSpatialForce<double>>>>());
  this->DeclarePeriodicUnrestrictedUpdateEvent(
      kGripperLcmPeriod, 0.,
      &QPFingersControlDecoder::UpdateAbstractState);
  // Register a forced discrete state update event. It is added for unit test,
  // or for potential users who require forced updates.
  this->DeclareForcedUnrestrictedUpdateEvent(
      &QPFingersControlDecoder::UpdateAbstractState);
}

systems::EventStatus QPFingersControlDecoder::UpdateAbstractState(
    const Context<double>& context, State<double>* state) const {
  lcmt_planar_manipuland_spatial_forces spatial_forces_lcm =
      this->GetInputPort("spatial_forces_lcm")
          .Eval<lcmt_planar_manipuland_spatial_forces>(context);

  auto& qp_fingers_control =
      state->get_mutable_abstract_state<std::unordered_map<
          Finger, multibody::ExternallyAppliedSpatialForce<double>>>(0);
  qp_fingers_control.clear();

  if (spatial_forces_lcm.num_forces > 0) {
    DRAKE_DEMAND(spatial_forces_lcm.manip_body_name == "brick_link");
    DRAKE_DEMAND(static_cast<int>(spatial_forces_lcm.num_forces) ==
                 static_cast<int>(spatial_forces_lcm.forces.size()));
    for (auto& spatial_force_lcm : spatial_forces_lcm.forces) {
      DRAKE_DEMAND(spatial_force_lcm.manip_body_name == "brick_link");
      multibody::ExternallyAppliedSpatialForce<double> applied_spatial_force;
      applied_spatial_force.body_index = brick_body_index_;
      applied_spatial_force.p_BoBq_B(0) = 0;                              // x
      applied_spatial_force.p_BoBq_B(1) = spatial_force_lcm.p_BoBq_B[0];  // y
      applied_spatial_force.p_BoBq_B(2) = spatial_force_lcm.p_BoBq_B[1];  // z

      Eigen::Vector3d force(0, spatial_force_lcm.force_Bq_W[0],
                            spatial_force_lcm.force_Bq_W[1]);
      Eigen::Vector3d torque = Eigen::Vector3d::Zero();
      torque(0) = spatial_force_lcm.torque_Bq_W;  // tau_x
      applied_spatial_force.F_Bq_W =
          multibody::SpatialForce<double>(torque, force);
      qp_fingers_control.emplace(to_Finger(spatial_force_lcm.finger_name),
                                 applied_spatial_force);
    }
  }
  return systems::EventStatus::Succeeded();
}

void QPFingersControlDecoder::OutputFingersControl(
    const systems::Context<double>& context,
    std::unordered_map<Finger,
                       multibody::ExternallyAppliedSpatialForce<double>>*
        fingers_control) const {
  fingers_control->clear();
  *fingers_control = context.get_abstract_state<std::unordered_map<
      Finger, multibody::ExternallyAppliedSpatialForce<double>>>(0);
}

QPFingersControlEncoder::QPFingersControlEncoder() {
  this->DeclareAbstractInputPort(
      "qp_fingers_control",
      Value<std::unordered_map<Finger, multibody::ExternallyAppliedSpatialForce<
                                           double>>>{});
  this->DeclareAbstractOutputPort(
      "spatial_forces_lcm", &QPFingersControlEncoder::EncodeFingersControl);
}

void QPFingersControlEncoder::EncodeFingersControl(
    const drake::systems::Context<double>& context,
    drake::lcmt_planar_manipuland_spatial_forces* spatial_forces_lcm) const {
  std::string brick_body_name = "brick_link";  // hard-coded for now.
  spatial_forces_lcm->utime = static_cast<int64_t>(context.get_time() * 1e6);
  spatial_forces_lcm->manip_body_name = brick_body_name;
  std::unordered_map<Finger, multibody::ExternallyAppliedSpatialForce<double>>
      qp_fingers_control =
          this->GetInputPort("qp_fingers_control")
              .Eval<std::unordered_map<
                  Finger, multibody::ExternallyAppliedSpatialForce<double>>>(
                  context);

  spatial_forces_lcm->num_forces = qp_fingers_control.size();
  spatial_forces_lcm->forces.clear();
  for (auto& finger_force : qp_fingers_control) {
    lcmt_planar_manipuland_spatial_force spatial_force_lcm;
    spatial_force_lcm.utime = spatial_forces_lcm->utime;
    spatial_force_lcm.manip_body_name = brick_body_name;
    spatial_force_lcm.finger_name = to_string(finger_force.first);
    spatial_force_lcm.p_BoBq_B[0] = finger_force.second.p_BoBq_B(1);  // y
    spatial_force_lcm.p_BoBq_B[1] = finger_force.second.p_BoBq_B(2);  // z
    spatial_force_lcm.force_Bq_W[0] =
        finger_force.second.F_Bq_W.translational()(1);  // fy
    spatial_force_lcm.force_Bq_W[1] =
        finger_force.second.F_Bq_W.translational()(2);  // fz
    spatial_force_lcm.torque_Bq_W = finger_force.second.F_Bq_W.rotational()(0);  // tx.
    spatial_forces_lcm->forces.push_back(spatial_force_lcm);
  }
}

QPEstimatedStateDecoder::QPEstimatedStateDecoder(const int num_plant_states)
    : num_plant_states_(num_plant_states) {
  this->DeclareAbstractInputPort("planar_plant_state_lcm",
                                 Value<lcmt_planar_plant_state>{});
  this->DeclareVectorOutputPort("qp_estimated_plant_state",
                                systems::BasicVector<double>(num_plant_states),
                                &QPEstimatedStateDecoder::OutputEstimatedState);
  this->DeclarePeriodicDiscreteUpdateEvent(
      kGripperLcmPeriod, 0.,
      &QPEstimatedStateDecoder::UpdateDiscreteState);
  // Register a forced discrete state update event. It is added for unit test,
  // or for potential users who require forced updates.
  this->DeclareForcedDiscreteUpdateEvent(
      &QPEstimatedStateDecoder::UpdateDiscreteState);
  // Discrete state holds pos, vel, torque.
  this->DeclareDiscreteState(num_plant_states);
}

systems::EventStatus QPEstimatedStateDecoder::UpdateDiscreteState(
    const Context<double>& context,
    DiscreteValues<double>* discrete_state) const {
  const auto& planar_plant_state_lcm = this->GetInputPort("planar_plant_state_lcm")
                                 .Eval<lcmt_planar_plant_state>(context);

  // If we're using a default constructed message (haven't received
  // a command yet), keep using the initial state.
  Eigen::Ref<Eigen::VectorXd> plant_state =
      discrete_state->get_mutable_vector(0).get_mutable_value();

  DRAKE_DEMAND(plant_state.size() == num_plant_states_);
  DRAKE_DEMAND(planar_plant_state_lcm.num_states == 0 ||
      planar_plant_state_lcm.num_states == num_plant_states_);

  if (planar_plant_state_lcm.num_states == num_plant_states_) {
    std::copy(planar_plant_state_lcm.plant_state.data(),
              planar_plant_state_lcm.plant_state.data() + num_plant_states_,
              plant_state.data());
  }
  return systems::EventStatus::Succeeded();
}

void QPEstimatedStateDecoder::OutputEstimatedState(
    const drake::systems::Context<double>& context,
    systems::BasicVector<double>* plant_state) const {
  Eigen::VectorBlock<VectorX<double>> output_vec =
      plant_state->get_mutable_value();
  output_vec = context.get_discrete_state(0).get_value();
}

QPEstimatedStateEncoder::QPEstimatedStateEncoder(const int num_plant_states)
    : num_plant_states_(num_plant_states) {
  this->DeclareVectorInputPort("qp_estimated_plant_state",
                               systems::BasicVector<double>(num_plant_states));
  this->DeclareAbstractOutputPort(
      "planar_plant_state_lcm",
      &QPEstimatedStateEncoder::EncodeEstimatedState);
}

void QPEstimatedStateEncoder::EncodeEstimatedState(
    const drake::systems::Context<double>& context,
    drake::lcmt_planar_plant_state* planar_plant_state_lcm) const {
  planar_plant_state_lcm->utime = static_cast<int64_t>(context.get_time() * 1e6);
  planar_plant_state_lcm->num_states = num_plant_states_;
  VectorX<double> estimated_plant_state =
      this->EvalVectorInput(
              context,
              this->GetInputPort("qp_estimated_plant_state").get_index())
          ->get_value();
  planar_plant_state_lcm->plant_state.resize(num_plant_states_);
  for (int i = 0; i < num_plant_states_; i++) {
    planar_plant_state_lcm->plant_state[i] = estimated_plant_state(i);
  }
}

QPFingerFaceAssignmentsDecoder::QPFingerFaceAssignmentsDecoder() {
  this->DeclareAbstractInputPort(
      "finger_face_assignments_lcm",
      Value<lcmt_planar_gripper_finger_face_assignments>{});
  this->DeclareAbstractOutputPort(
      "qp_finger_face_assignments",
      &QPFingerFaceAssignmentsDecoder::OutputFingerFaceAssignments);

  // State holds an abstract value of
  // std::unordered_map<Finger, std::pair(BrickFace, Eigen::Vector2d)>
  this->DeclareAbstractState(
      std::make_unique<Value<std::unordered_map<
          Finger, std::pair<BrickFace, Eigen::Vector2d>>>>());
  this->DeclarePeriodicUnrestrictedUpdateEvent(
      kGripperLcmPeriod, 0.,
      &QPFingerFaceAssignmentsDecoder::UpdateAbstractState);
  // Register a forced discrete state update event. It is added for unit test,
  // or for potential users who require forced updates.
  this->DeclareForcedUnrestrictedUpdateEvent(
      &QPFingerFaceAssignmentsDecoder::UpdateAbstractState);
}

systems::EventStatus QPFingerFaceAssignmentsDecoder::UpdateAbstractState(
    const Context<double>& context, State<double>* state) const {
  auto assignments_lcm =
      this->GetInputPort("finger_face_assignments_lcm")
          .Eval<lcmt_planar_gripper_finger_face_assignments>(context);

  // If we've received at least the first lcm message, update the state here.
  if (assignments_lcm.num_fingers != -1) {
    DRAKE_DEMAND(
        static_cast<int>(assignments_lcm.finger_face_assignments.size()) ==
        assignments_lcm.num_fingers);
    auto& assignments = state->get_mutable_abstract_state<
        std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>>(0);
    assignments.clear();
    for (int i = 0; i < assignments_lcm.num_fingers; ++i) {
      auto assignment_lcm = assignments_lcm.finger_face_assignments[i];
      Finger finger = to_Finger(assignment_lcm.finger_name);
      BrickFace brick_face = to_BrickFace(assignment_lcm.brick_face_name);
      Eigen::Vector2d p_BoBq_B;
      p_BoBq_B(0) = assignment_lcm.p_BoBq_B[0];
      p_BoBq_B(1) = assignment_lcm.p_BoBq_B[1];
      assignments.emplace(finger, std::make_pair(brick_face, p_BoBq_B));
    }
  }
  return systems::EventStatus::Succeeded();
}

void QPFingerFaceAssignmentsDecoder::OutputFingerFaceAssignments(
    const systems::Context<double>& context,
    std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>*
        finger_face_assignments) const {
  finger_face_assignments->clear();
  *finger_face_assignments = context.get_abstract_state<
      std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>>(0);
}

QPFingerFaceAssignmentsEncoder::QPFingerFaceAssignmentsEncoder() {
  this->DeclareAbstractInputPort(
      "qp_finger_face_assignments",
      Value<
          std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>>{});

  this->DeclareAbstractOutputPort(
      "finger_face_assignments_lcm",
      &QPFingerFaceAssignmentsEncoder::EncodeFingerFaceAssignments);
}

void QPFingerFaceAssignmentsEncoder::EncodeFingerFaceAssignments(
    const drake::systems::Context<double>& context,
    lcmt_planar_gripper_finger_face_assignments*
        finger_face_assignments_lcm) const {
  auto finger_face_assignments =
      this->GetInputPort("qp_finger_face_assignments")
          .Eval<std::unordered_map<Finger,
                                   std::pair<BrickFace, Eigen::Vector2d>>>(
              context);
  size_t num_fingers = finger_face_assignments.size();
  finger_face_assignments_lcm->num_fingers = num_fingers;
//  finger_face_assignments_lcm->finger_face_assignments.resize(num_fingers);
  finger_face_assignments_lcm->finger_face_assignments.clear();
  finger_face_assignments_lcm->utime =
      static_cast<int64_t>(context.get_time() * 1e6);
  for (auto& finger_face_assignment : finger_face_assignments) {
    lcmt_planar_gripper_finger_face_assignment finger_face_assignment_lcm;
    finger_face_assignment_lcm.utime = finger_face_assignments_lcm->utime;
    finger_face_assignment_lcm.finger_name =
        to_string(finger_face_assignment.first);
    finger_face_assignment_lcm.brick_face_name =
        to_string(finger_face_assignment.second.first);
    finger_face_assignment_lcm.p_BoBq_B[0] =
        finger_face_assignment.second.second(0);
    finger_face_assignment_lcm.p_BoBq_B[1] =
        finger_face_assignment.second.second(1);
    finger_face_assignments_lcm->finger_face_assignments.push_back(
        finger_face_assignment_lcm);
  }
  DRAKE_DEMAND(finger_face_assignments_lcm->finger_face_assignments.size() ==
               num_fingers);
}

QPBrickDesiredDecoder::QPBrickDesiredDecoder(const int num_brick_states,
                                             const int num_brick_accels)
    : num_brick_states_(num_brick_states), num_brick_accels_(num_brick_accels) {
  this->DeclareAbstractInputPort("brick_desired_lcm",
                                 Value<lcmt_planar_manipuland_desired>{});
  this->DeclareVectorOutputPort(
      "qp_desired_brick_state", systems::BasicVector<double>(num_brick_states_),
      &QPBrickDesiredDecoder::OutputBrickDesiredState);
  this->DeclareVectorOutputPort(
      "qp_desired_brick_accel", systems::BasicVector<double>(num_brick_accels_),
      &QPBrickDesiredDecoder::DecodeBrickDesiredAccel);

  this->DeclarePeriodicDiscreteUpdateEvent(
      kGripperLcmPeriod, 0.,
      &QPBrickDesiredDecoder::UpdateDiscreteState);
  // Register a forced discrete state update event. It is added for unit test,
  // or for potential users who require forced updates.
  this->DeclareForcedDiscreteUpdateEvent(
      &QPBrickDesiredDecoder::UpdateDiscreteState);
  // Discrete state holds brick state.
  brick_state_index_ = this->DeclareDiscreteState(num_brick_states_);
  // Discrete state holds brick accels.
  brick_accel_index_ = this->DeclareDiscreteState(num_brick_accels_);
}

systems::EventStatus QPBrickDesiredDecoder::UpdateDiscreteState(
    const Context<double>& context,
    DiscreteValues<double>* discrete_state) const {
  const auto& brick_desired_lcm =
      this->GetInputPort("brick_desired_lcm")
          .Eval<lcmt_planar_manipuland_desired>(context);

  // If we're using a default constructed message (haven't received
  // a command yet), keep using the initial state.

  // Brick state is: {y, z theta, ydot, zdot, thetadot}.
  Eigen::Ref<Eigen::VectorXd> brick_state =
      discrete_state->get_mutable_vector(brick_state_index_)
          .get_mutable_value();
  // Brick accel is: {ay, az, thetaddot}.
  Eigen::Ref<Eigen::VectorXd> brick_accel =
      discrete_state->get_mutable_vector(brick_accel_index_)
          .get_mutable_value();

  DRAKE_DEMAND(brick_state.size() == num_brick_states_);
  DRAKE_DEMAND(brick_accel.size() == num_brick_accels_);

  DRAKE_DEMAND(brick_desired_lcm.num_states == 0 ||
      brick_desired_lcm.num_states == num_brick_states_);
  DRAKE_DEMAND(brick_desired_lcm.num_accels == 0 ||
      brick_desired_lcm.num_accels == num_brick_accels_);

  if (brick_desired_lcm.num_states == num_brick_states_) {
    DRAKE_DEMAND(static_cast<int>(brick_desired_lcm.desired_state.size()) ==
        num_brick_states_);
    DRAKE_DEMAND(static_cast<int>(brick_desired_lcm.desired_accel.size()) ==
        num_brick_accels_);
    std::copy(brick_desired_lcm.desired_state.data(),
              brick_desired_lcm.desired_state.data() + num_brick_states_,
              brick_state.data());
    std::copy(brick_desired_lcm.desired_accel.data(),
              brick_desired_lcm.desired_accel.data() + num_brick_accels_,
              brick_accel.data());
  }
  return systems::EventStatus::Succeeded();
}

void QPBrickDesiredDecoder::OutputBrickDesiredState(
    const systems::Context<double>& context,
    systems::BasicVector<double>* qp_desired_brick_state) const {
  Eigen::VectorBlock<VectorX<double>> output_vec =
      qp_desired_brick_state->get_mutable_value();
  output_vec = context.get_discrete_state(brick_state_index_).get_value();
}

void QPBrickDesiredDecoder::DecodeBrickDesiredAccel(
    const systems::Context<double>& context,
    systems::BasicVector<double>* qp_desired_brick_accel) const {
  Eigen::VectorBlock<VectorX<double>> output_vec =
      qp_desired_brick_accel->get_mutable_value();
  output_vec = context.get_discrete_state(brick_accel_index_).get_value();
}

QPBrickDesiredEncoder::QPBrickDesiredEncoder(const int num_brick_states,
                                             const int num_brick_accels)
    : num_brick_states_(num_brick_states),
      num_brick_accels_(num_brick_accels) {
  this->DeclareVectorInputPort("qp_desired_brick_state",
                               systems::BasicVector<double>(num_brick_states_));
  this->DeclareVectorInputPort("qp_desired_brick_accel",
                               systems::BasicVector<double>(num_brick_accels_));
  this->DeclareAbstractOutputPort(
      "brick_desired_lcm",
      &QPBrickDesiredEncoder::EncodeBrickDesired);
}

void QPBrickDesiredEncoder::EncodeBrickDesired(
    const drake::systems::Context<double>& context,
    drake::lcmt_planar_manipuland_desired* planar_brick_desired_lcm) const {
  planar_brick_desired_lcm->utime =
      static_cast<int64_t>(context.get_time() * 1e6);
  planar_brick_desired_lcm->num_states = num_brick_states_;
  planar_brick_desired_lcm->num_accels = num_brick_accels_;
  Eigen::VectorXd desired_brick_state(num_brick_states_);
  Eigen::VectorXd desired_brick_accel(num_brick_accels_);
  desired_brick_state = this->GetInputPort("qp_desired_brick_state")
                            .Eval<systems::BasicVector<double>>(context)
                            .get_value();
  DRAKE_DEMAND(desired_brick_state.size() == num_brick_states_);
  desired_brick_accel = this->GetInputPort("qp_desired_brick_accel")
                            .Eval<systems::BasicVector<double>>(context)
                            .get_value();
  DRAKE_DEMAND(desired_brick_accel.size() == num_brick_accels_);

  planar_brick_desired_lcm->desired_state.resize(num_brick_states_);
  std::copy(desired_brick_state.data(),
            desired_brick_state.data() + num_brick_states_,
            planar_brick_desired_lcm->desired_state.data());
  planar_brick_desired_lcm->desired_accel.resize(num_brick_accels_);
  std::copy(desired_brick_accel.data(),
            desired_brick_accel.data() + num_brick_accels_,
            planar_brick_desired_lcm->desired_accel.data());
}

PlanarGripperQPControllerLCM::PlanarGripperQPControllerLCM(
    const int num_multibody_states, const int num_brick_states,
    const int num_brick_accels, const multibody::BodyIndex brick_index,
    drake::lcm::DrakeLcmInterface* lcm, double publish_period) {
  systems::DiagramBuilder<double> builder;

  /*
   * The LCM (subscribe) inputs the the local simulation, which are received
   * from the outputs (publishers) of the remote QP controller.
   */
  auto qp_fingers_control_sub =
      builder.AddSystem(systems::lcm::LcmSubscriberSystem::Make<
                        drake::lcmt_planar_manipuland_spatial_forces>(
          "QP_FINGERS_CONTROL", lcm));
  auto qp_fingers_control_dec =
      builder.AddSystem<QPFingersControlDecoder>(brick_index);
  builder.Connect(qp_fingers_control_sub->get_output_port(),
                  qp_fingers_control_dec->GetInputPort("spatial_forces_lcm"));
  builder.ExportOutput(
      qp_fingers_control_dec->GetOutputPort("qp_fingers_control"),
      "qp_fingers_control");

  auto qp_brick_control_sub =
      builder.AddSystem(systems::lcm::LcmSubscriberSystem::Make<
          drake::lcmt_planar_manipuland_spatial_forces>(
          "QP_BRICK_CONTROL", lcm));
  auto qp_brick_control_dec =
      builder.AddSystem<QPBrickControlDecoder>(brick_index);
  builder.Connect(qp_brick_control_sub->get_output_port(),
                  qp_brick_control_dec->GetInputPort("spatial_forces_lcm"));
  builder.ExportOutput(
      qp_brick_control_dec->GetOutputPort("qp_brick_control"),
      "qp_brick_control");

  /*
   * The LCM (publish) outputs from the local simulation, which are sent to the
   * inputs (subscribers) of the remote QP controller.
   */
  auto qp_estimated_plant_state_pub = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<drake::lcmt_planar_plant_state>(
          "QP_ESTIMATED_PLANT_STATE", lcm, publish_period));
  auto qp_estimated_plant_state_enc =
      builder.AddSystem<QPEstimatedStateEncoder>(num_multibody_states);
  builder.Connect(
      qp_estimated_plant_state_enc->GetOutputPort("planar_plant_state_lcm"),
      qp_estimated_plant_state_pub->get_input_port());
  builder.ExportInput(
      qp_estimated_plant_state_enc->GetInputPort("qp_estimated_plant_state"),
      "qp_estimated_plant_state");

  auto qp_finger_face_assignments_pub =
      builder.AddSystem(systems::lcm::LcmPublisherSystem::Make<
                        drake::lcmt_planar_gripper_finger_face_assignments>(
          "QP_FINGER_FACE_ASSIGNMENTS", lcm, publish_period));
  auto qp_finger_face_assignments_enc =
      builder.AddSystem<QPFingerFaceAssignmentsEncoder>();
  builder.Connect(qp_finger_face_assignments_enc->GetOutputPort(
                      "finger_face_assignments_lcm"),
                  qp_finger_face_assignments_pub->get_input_port());
  builder.ExportInput(qp_finger_face_assignments_enc->GetInputPort(
                          "qp_finger_face_assignments"),
                      "qp_finger_face_assignments");

  auto qp_brick_desired_pub =
      builder.AddSystem(systems::lcm::LcmPublisherSystem::Make<
                        drake::lcmt_planar_manipuland_desired>(
          "QP_BRICK_DESIRED", lcm, publish_period));
  auto qp_brick_desired_enc = builder.AddSystem<QPBrickDesiredEncoder>(
      num_brick_states, num_brick_accels);
  builder.Connect(qp_brick_desired_enc->GetOutputPort("brick_desired_lcm"),
                  qp_brick_desired_pub->get_input_port());
  builder.ExportInput(qp_brick_desired_enc->GetInputPort(
      "qp_desired_brick_state"), "qp_desired_brick_state");
  builder.ExportInput(qp_brick_desired_enc->GetInputPort(
      "qp_desired_brick_accel"), "qp_desired_brick_accel");

  builder.BuildInto(this);
}

PlanarGripperSimulationLCM::PlanarGripperSimulationLCM(
    const int num_multibody_states, const int num_brick_states,
    const int num_brick_accels, drake::lcm::DrakeLcmInterface* lcm,
    double publish_period) {
  systems::DiagramBuilder<double> builder;
  /*
   * The LCM (subscribe) inputs to the local QP controller, which are received
   * from the outputs (publishers) of the remote simulation.
   */
  qp_estimated_plant_state_sub_ = builder.AddSystem(
      systems::lcm::LcmSubscriberSystem::Make<drake::lcmt_planar_plant_state>(
          "QP_ESTIMATED_PLANT_STATE", lcm));
  auto qp_estimated_plant_state_dec =
      builder.AddSystem<QPEstimatedStateDecoder>(num_multibody_states);
  builder.Connect(
      qp_estimated_plant_state_sub_->get_output_port(),
      qp_estimated_plant_state_dec->GetInputPort("planar_plant_state_lcm"));
  builder.ExportOutput(
      qp_estimated_plant_state_dec->GetOutputPort("qp_estimated_plant_state"),
      "qp_estimated_plant_state");

  qp_finger_face_assignments_sub_ =
      builder.AddSystem(systems::lcm::LcmSubscriberSystem::Make<
                        drake::lcmt_planar_gripper_finger_face_assignments>(
          "QP_FINGER_FACE_ASSIGNMENTS", lcm));
  auto qp_finger_face_assignments_dec =
      builder.AddSystem<QPFingerFaceAssignmentsDecoder>();
  builder.Connect(qp_finger_face_assignments_sub_->get_output_port(),
                  qp_finger_face_assignments_dec->GetInputPort(
                      "finger_face_assignments_lcm"));
  builder.ExportOutput(qp_finger_face_assignments_dec->GetOutputPort(
                           "qp_finger_face_assignments"),
                       "qp_finger_face_assignments");

  qp_brick_desired_sub_ =
      builder.AddSystem(systems::lcm::LcmSubscriberSystem::Make<
          drake::lcmt_planar_manipuland_desired>(
          "QP_BRICK_DESIRED", lcm));
  auto qp_brick_desired_dec = builder.AddSystem<QPBrickDesiredDecoder>(
      num_brick_states, num_brick_accels);
  builder.Connect(qp_brick_desired_sub_->get_output_port(),
                  qp_brick_desired_dec->GetInputPort("brick_desired_lcm"));
  builder.ExportOutput(qp_brick_desired_dec->GetOutputPort(
      "qp_desired_brick_state"),
                       "qp_desired_brick_state");
  builder.ExportOutput(qp_brick_desired_dec->GetOutputPort(
      "qp_desired_brick_accel"),
                       "qp_desired_brick_accel");

  /*
   * The LCM (publish) outputs of the local QP controller, which are sent to the
   * inputs (subscribers) of the remote simulation.
   */
  auto qp_fingers_control_pub =
      builder.AddSystem(systems::lcm::LcmPublisherSystem::Make<
                        drake::lcmt_planar_manipuland_spatial_forces>(
          "QP_FINGERS_CONTROL", lcm, publish_period));
  auto qp_fingers_control_enc =
      builder.AddSystem<QPFingersControlEncoder>();
  builder.Connect(
      qp_fingers_control_enc->GetOutputPort("spatial_forces_lcm"),
      qp_fingers_control_pub->get_input_port());
  builder.ExportInput(
      qp_fingers_control_enc->GetInputPort("qp_fingers_control"),
      "qp_fingers_control");

  auto qp_brick_control_pub =
      builder.AddSystem(systems::lcm::LcmPublisherSystem::Make<
          drake::lcmt_planar_manipuland_spatial_forces>(
          "QP_BRICK_CONTROL", lcm, publish_period));
  auto qp_brick_control_enc =
      builder.AddSystem<QPBrickControlEncoder>();
  builder.Connect(
      qp_brick_control_enc->GetOutputPort("spatial_forces_lcm"),
      qp_brick_control_pub->get_input_port());
  builder.ExportInput(
      qp_brick_control_enc->GetInputPort("qp_brick_control"),
      "qp_brick_control");

  builder.BuildInto(this);
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
