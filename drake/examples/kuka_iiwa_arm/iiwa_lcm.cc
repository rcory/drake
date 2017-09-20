#include "drake/examples/kuka_iiwa_arm/iiwa_lcm.h"

#include <vector>

#include "drake/common/drake_assert.h"
#include "drake/lcmt_iiwa_command.hpp"
#include "drake/lcmt_iiwa_status.hpp"
#include "external/optitrack_driver/lcmtypes/optitrack/optitrack_rigid_body_t.hpp"
#include "drake/multibody/rigid_body_plant/kinematics_results.h"
#include "drake/examples/kuka_iiwa_arm/dev/box_rotation/optitrack_sim.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {

using systems::BasicVector;
using systems::Context;
using systems::DiscreteValues;
using systems::State;
using systems::SystemOutput;
using systems::DiscreteUpdateEvent;

// This value is chosen to match the value in getSendPeriodMilliSec()
// when initializing the FRI configuration on the iiwa's control
// cabinet.
const double kIiwaLcmStatusPeriod = 0.005;

IiwaCommandReceiver::IiwaCommandReceiver(int num_joints)
    : num_joints_(num_joints) {
  this->DeclareAbstractInputPort();
  this->DeclareVectorOutputPort(systems::BasicVector<double>(num_joints_ * 2),
                                &IiwaCommandReceiver::OutputCommand);
  this->DeclarePeriodicDiscreteUpdate(kIiwaLcmStatusPeriod);
  this->DeclareDiscreteState(num_joints_ * 2);
}

void IiwaCommandReceiver::set_initial_position(
    Context<double>* context,
    const Eigen::Ref<const VectorX<double>> x) const {
  auto state_value =
      context->get_mutable_discrete_state(0)->get_mutable_value();
  DRAKE_ASSERT(x.size() == num_joints_);
  state_value.head(num_joints_) = x;
  state_value.tail(num_joints_) = VectorX<double>::Zero(num_joints_);
}

void IiwaCommandReceiver::DoCalcDiscreteVariableUpdates(
    const Context<double>& context,
    const std::vector<const DiscreteUpdateEvent<double>*>&,
    DiscreteValues<double>* discrete_state) const {
  const systems::AbstractValue* input = this->EvalAbstractInput(context, 0);
  DRAKE_ASSERT(input != nullptr);
  const auto& command = input->GetValue<lcmt_iiwa_command>();
  // TODO(sam.creasey) Support torque control.
  DRAKE_ASSERT(command.num_torques == 0);


  // If we're using a default constructed message (haven't received
  // a command yet), keep using the initial state.
  if (command.num_joints != 0) {
    DRAKE_DEMAND(command.num_joints == num_joints_);
    VectorX<double> new_positions(num_joints_);
    for (int i = 0; i < command.num_joints; ++i) {
      new_positions(i) = command.joint_position[i];
    }

    BasicVector<double>* state = discrete_state->get_mutable_vector(0);
    auto state_value = state->get_mutable_value();
    state_value.tail(num_joints_) =
        (new_positions - state_value.head(num_joints_)) / kIiwaLcmStatusPeriod;
    state_value.head(num_joints_) = new_positions;
  }
}

void IiwaCommandReceiver::OutputCommand(const Context<double>& context,
                                        BasicVector<double>* output) const {
  Eigen::VectorBlock<VectorX<double>> output_vec =
      output->get_mutable_value();
  output_vec = context.get_discrete_state(0)->get_value();
}

IiwaCommandSender::IiwaCommandSender(int num_joints)
    : num_joints_(num_joints),
      position_input_port_(
          this->DeclareInputPort(
              systems::kVectorValued, num_joints_).get_index()),
      torque_input_port_(
          this->DeclareInputPort(
              systems::kVectorValued, num_joints_).get_index()) {
  this->DeclareAbstractOutputPort(&IiwaCommandSender::OutputCommand);
}

void IiwaCommandSender::OutputCommand(
    const Context<double>& context, lcmt_iiwa_command* output) const {
  lcmt_iiwa_command& command = *output;

  command.utime = context.get_time() * 1e6;
  const systems::BasicVector<double>* positions =
      this->EvalVectorInput(context, 0);

  command.num_joints = num_joints_;
  command.joint_position.resize(num_joints_);
  for (int i = 0; i < num_joints_; ++i) {
    command.joint_position[i] = positions->GetAtIndex(i);
  }

  const systems::BasicVector<double>* torques =
      this->EvalVectorInput(context, 1);
  if (torques == nullptr) {
    command.num_torques = 0;
    command.joint_torque.clear();
  } else {
    command.num_torques = num_joints_;
    command.joint_torque.resize(num_joints_);
    for (int i = 0; i < num_joints_; ++i) {
      command.joint_torque[i] = torques->GetAtIndex(i);
    }
  }
}

IiwaStatusReceiver::IiwaStatusReceiver(int num_joints)
    : num_joints_(num_joints),
      measured_position_output_port_(
          this->DeclareVectorOutputPort(
                  systems::BasicVector<double>(num_joints_ * 2),
                  &IiwaStatusReceiver::OutputMeasuredPosition)
              .get_index()),
      commanded_position_output_port_(
          this->DeclareVectorOutputPort(
                  systems::BasicVector<double>(num_joints_),
                  &IiwaStatusReceiver::OutputCommandedPosition)
              .get_index()) {
  this->DeclareAbstractInputPort();
  this->DeclareDiscreteState(num_joints_ * 3);
  this->DeclarePeriodicDiscreteUpdate(kIiwaLcmStatusPeriod);
}

void IiwaStatusReceiver::DoCalcDiscreteVariableUpdates(
    const Context<double>& context,
    const std::vector<const DiscreteUpdateEvent<double>*>&,
    DiscreteValues<double>* discrete_state) const {
  const systems::AbstractValue* input = this->EvalAbstractInput(context, 0);
  DRAKE_ASSERT(input != nullptr);
  const auto& status = input->GetValue<lcmt_iiwa_status>();

  // If we're using a default constructed message (haven't received
  // status yet), keep using the initial state.
  if (status.num_joints != 0) {
    DRAKE_DEMAND(status.num_joints == num_joints_);

    VectorX<double> measured_position(num_joints_);
    VectorX<double> commanded_position(num_joints_);
    for (int i = 0; i < status.num_joints; ++i) {
      measured_position(i) = status.joint_position_measured[i];
      commanded_position(i) = status.joint_position_commanded[i];
    }

    BasicVector<double>* state = discrete_state->get_mutable_vector(0);
    auto state_value = state->get_mutable_value();
    state_value.segment(num_joints_, num_joints_) =
        (measured_position - state_value.head(num_joints_)) /
        kIiwaLcmStatusPeriod;
    state_value.head(num_joints_) = measured_position;
    state_value.tail(num_joints_) = commanded_position;
  }
}

void IiwaStatusReceiver::OutputMeasuredPosition(const Context<double>& context,
                                       BasicVector<double>* output) const {
  const auto state_value = context.get_discrete_state(0)->get_value();

  Eigen::VectorBlock<VectorX<double>> measured_position_output =
      output->get_mutable_value();
  measured_position_output = state_value.head(num_joints_ * 2);
}

void IiwaStatusReceiver::OutputCommandedPosition(
    const Context<double>& context, BasicVector<double>* output) const {
  const auto state_value = context.get_discrete_state(0)->get_value();

  Eigen::VectorBlock<VectorX<double>> commanded_position_output =
      output->get_mutable_value();
  commanded_position_output = state_value.tail(num_joints_);
}

IiwaStatusSender::IiwaStatusSender(int num_joints)
    : num_joints_(num_joints) {
  this->DeclareInputPort(systems::kVectorValued, num_joints_ * 2);
  this->DeclareInputPort(systems::kVectorValued, num_joints_ * 2);
  this->DeclareAbstractOutputPort(&IiwaStatusSender::MakeOutputStatus,
                                  &IiwaStatusSender::OutputStatus);
}

lcmt_iiwa_status IiwaStatusSender::MakeOutputStatus() const {
  lcmt_iiwa_status msg{};
  msg.num_joints = num_joints_;
  msg.joint_position_measured.resize(msg.num_joints, 0);
  msg.joint_position_commanded.resize(msg.num_joints, 0);
  msg.joint_position_ipo.resize(msg.num_joints, 0);
  msg.joint_torque_measured.resize(msg.num_joints, 0);
  msg.joint_torque_commanded.resize(msg.num_joints, 0);
  msg.joint_torque_external.resize(msg.num_joints, 0);
  return msg;
}

void IiwaStatusSender::OutputStatus(
    const Context<double>& context, lcmt_iiwa_status* output) const {
  lcmt_iiwa_status& status = *output;

  status.utime = context.get_time() * 1e6;
  const systems::BasicVector<double>* command =
      this->EvalVectorInput(context, 0);
  const systems::BasicVector<double>* state =
      this->EvalVectorInput(context, 1);
  for (int i = 0; i < num_joints_; ++i) {
    status.joint_position_measured[i] = state->GetAtIndex(i);
    status.joint_position_commanded[i] = command->GetAtIndex(i);
  }
}

/// optitrack stuff
using drake::examples::kuka_iiwa_arm::box_rotation::TrackedObject;

OptitrackFrameSender::OptitrackFrameSender() {
  this->DeclareAbstractInputPort(); // of type std::vector<TrackedObjects>
  this->DeclareAbstractOutputPort(&OptitrackFrameSender::MakeOutputStatus,
                                  &OptitrackFrameSender::OutputStatus);
}

optitrack::optitrack_frame_t OptitrackFrameSender::MakeOutputStatus() const {
  optitrack::optitrack_frame_t msg{};

  // TODO(rcory): Can I initialize the number of rigid bodies from
  // the number of OptiTrack sim bodies??
  msg.num_rigid_bodies = 3;
  msg.rigid_bodies.resize(3);

  return msg;
}

void OptitrackFrameSender::OutputStatus(
    const Context<double>& context, optitrack::optitrack_frame_t* output) const {
  optitrack::optitrack_frame_t& status = *output;

  status.utime = context.get_time() * 1e6;

  const std::vector<TrackedObject>* mocap_objects =
      this->EvalInputValue<std::vector<TrackedObject>>(context,0);

  for (size_t i = 0; i < mocap_objects->size(); ++i) {
    status.rigid_bodies[i].id = (*mocap_objects)[i].id;

    status.rigid_bodies[i].xyz[0] = (float) (*mocap_objects)[i].trans[0];
    status.rigid_bodies[i].xyz[1] = (float) (*mocap_objects)[i].trans[1];
    status.rigid_bodies[i].xyz[2] = (float) (*mocap_objects)[i].trans[2];

    status.rigid_bodies[i].quat[0] = (float) (*mocap_objects)[i].rot.x();
    status.rigid_bodies[i].quat[1] = (float) (*mocap_objects)[i].rot.y();
    status.rigid_bodies[i].quat[2] = (float) (*mocap_objects)[i].rot.z();
    status.rigid_bodies[i].quat[3] = (float) (*mocap_objects)[i].rot.w();
  }
}


}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
