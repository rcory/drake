#pragma once

/// @file
/// This file contains classes dealing with sending/receiving UDP messages
/// related to the planar gripper.

#include <netinet/in.h>

#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace examples {
namespace planar_gripper {
// This is rather arbitrary, for now.
constexpr double kGripperUdpStatusPeriod = 0.002;

struct UdpMessage {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(UdpMessage)

  UdpMessage() {}

  virtual ~UdpMessage() {}

  int GetMessageSize() const { return DoGetMessageSize(); }

  void Deserialize(const uint8_t* msg) { DoDeserialize(msg); }
  void Serialize(uint8_t* msg) const { DoSerialize(msg); }

 private:
  virtual int DoGetMessageSize() const = 0;

  virtual void DoDeserialize(const uint8_t* msg) = 0;

  virtual void DoSerialize(uint8_t* msg) const = 0;
};

struct FingerFaceAssignment : public UdpMessage {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FingerFaceAssignment)

  FingerFaceAssignment() {}

  virtual ~FingerFaceAssignment() {}

  uint32_t utime;

  Finger finger;
  BrickFace brick_face;

  Eigen::Vector2d p_BoBq_B;

 private:
  virtual int DoGetMessageSize() const final;

  virtual void DoDeserialize(const uint8_t* msg) final;

  virtual void DoSerialize(uint8_t* msg) const final;
};

bool operator==(const FingerFaceAssignment& f1, const FingerFaceAssignment& f2);

struct FingerFaceAssignments : public UdpMessage {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FingerFaceAssignments)

  FingerFaceAssignments(int m_num_fingers)
      : num_fingers{static_cast<uint32_t>(m_num_fingers)},
        finger_face_assignments(m_num_fingers),
        in_contact(m_num_fingers, false) {}

  FingerFaceAssignments() : FingerFaceAssignments(0) {}

  virtual ~FingerFaceAssignments() {}

  uint32_t utime;

  uint32_t num_fingers;
  std::vector<FingerFaceAssignment> finger_face_assignments;
  std::vector<bool> in_contact;

 private:
  virtual int DoGetMessageSize() const final;

  virtual void DoDeserialize(const uint8_t* msg) final;

  virtual void DoSerialize(uint8_t* msg) const final;
};

bool operator==(const FingerFaceAssignments& f1,
                const FingerFaceAssignments& f2);

struct PlanarManipulandDesired : public UdpMessage {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(PlanarManipulandDesired)
  PlanarManipulandDesired(int m_num_states, int m_num_accel)
      : num_states{static_cast<uint32_t>(m_num_states)},
        num_accels{static_cast<uint32_t>(m_num_accel)},
        desired_state(num_states),
        desired_accel(num_accels) {}

  ~PlanarManipulandDesired() {}

  uint32_t utime;
  uint32_t num_states;
  uint32_t num_accels;
  Eigen::VectorXd desired_state;
  Eigen::VectorXd desired_accel;

 private:
  virtual int DoGetMessageSize() const final;

  virtual void DoDeserialize(const uint8_t* msg) final;

  virtual void DoSerialize(uint8_t* msg) const final;
};

bool operator==(const PlanarManipulandDesired& f1,
                const PlanarManipulandDesired& f2);

struct PlanarManipulandSpatialForce : public UdpMessage {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(PlanarManipulandSpatialForce)

  PlanarManipulandSpatialForce() {}

  virtual ~PlanarManipulandSpatialForce() {}

  multibody::ExternallyAppliedSpatialForce<double> ToSpatialForce(
      multibody::BodyIndex body_index) const;

  uint32_t utime;
  Finger finger;
  // (y, z) position of point Bq in body frame B.
  Eigen::Vector2d p_BoBq_B;

  // (y, z) force applied to body B at point Bq, expressed in the world frame W.
  Eigen::Vector2d force_Bq_W;

  // torque applied to body B at point Bq, expressed in the world frame W.
  double torque_Bq_W;

 private:
  virtual int DoGetMessageSize() const final;

  virtual void DoDeserialize(const uint8_t* msg) final;

  virtual void DoSerialize(uint8_t* msg) const final;
};

bool operator==(const PlanarManipulandSpatialForce& f1,
                const PlanarManipulandSpatialForce& f2);

struct PlanarManipulandSpatialForces : public UdpMessage {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(PlanarManipulandSpatialForces)

  PlanarManipulandSpatialForces(int m_num_forces);

  PlanarManipulandSpatialForces();

  ~PlanarManipulandSpatialForces() {}

  uint32_t utime;

  uint32_t num_forces;
  std::vector<PlanarManipulandSpatialForce> forces;
  std::vector<bool> in_contact;

 private:
  virtual int DoGetMessageSize() const final;

  virtual void DoDeserialize(const uint8_t* msg) final;

  virtual void DoSerialize(uint8_t* msg) const final;
};

bool operator==(const PlanarManipulandSpatialForces& f1,
                const PlanarManipulandSpatialForces& f2);

struct PlanarPlantState : public UdpMessage {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(PlanarPlantState)

  PlanarPlantState(int m_num_states);

  PlanarPlantState();

  virtual ~PlanarPlantState() {}

  uint32_t utime;
  uint32_t num_states;
  Eigen::VectorXd plant_state;

 private:
  virtual int DoGetMessageSize() const final;

  virtual void DoDeserialize(const uint8_t* msg) final;

  virtual void DoSerialize(uint8_t* msg) const final;
};

bool operator==(const PlanarPlantState& f1, const PlanarPlantState& f2);

/**
 * This system takes the simulation/hardware output as the input, and then
 * publish these inputs to UDP.
 */
class SimToQPUdpPublisherSystem : public systems::LeafSystem<double> {
 public:
  SimToQPUdpPublisherSystem(double publish_period, int local_port,
                            int remote_port, unsigned long remote_address,
                            int num_plant_states, int num_fingers,
                            int num_brick_states, int num_brick_accels);

  ~SimToQPUdpPublisherSystem();

  const systems::InputPort<double>& get_plant_state_input_port() const {
    return this->get_input_port(plant_state_input_port_);
  }

  const systems::InputPort<double>& get_finger_face_assignments_input_port()
      const {
    return this->get_input_port(finger_face_assignments_input_port_);
  }

  const systems::InputPort<double>& get_desired_brick_state_input_port() const {
    return this->get_input_port(desired_brick_state_input_port_);
  }

  const systems::InputPort<double>& get_desired_brick_accel_input_port() const {
    return this->get_input_port(desired_brick_accel_input_port_);
  }

 private:
  systems::EventStatus PublishInputAsUdpMessage(
      const systems::Context<double>& context) const;

  std::vector<uint8_t> Serialize(const systems::Context<double>& context) const;

  int file_descriptor_{};
  int remote_port_{};
  unsigned long remote_address_{};

  int num_plant_states_;
  int num_fingers_;
  int num_brick_states_;
  int num_brick_accels_;

  systems::InputPortIndex plant_state_input_port_;
  systems::InputPortIndex finger_face_assignments_input_port_;
  systems::InputPortIndex desired_brick_state_input_port_;
  systems::InputPortIndex desired_brick_accel_input_port_;
};

/**
 * This system takes the UDP message computed from QP controller, and then
 * outputs signals (qp_fingers_control and qp_brick_control) to
 * simulation/hardware.
 */
class QPControlUdpReceiverSystem : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QPControlUdpReceiverSystem)

  QPControlUdpReceiverSystem(int local_port, int num_fingers,
                             multibody::BodyIndex brick_body_index);
  ~QPControlUdpReceiverSystem();

  const systems::OutputPort<double>& get_qp_fingers_control_output_port()
      const {
    return this->get_output_port(qp_finger_control_output_port_);
  }

  const systems::OutputPort<double>& get_qp_brick_control_output_port() const {
    return this->get_output_port(qp_brick_control_output_port_);
  }

 private:
  systems::EventStatus ProcessMessageAndStoreToAbstractState(
      const systems::Context<double>& context,
      systems::State<double>* state) const;
  void OutputFingersControl(
      const systems::Context<double>& context,
      std::unordered_map<Finger,
                         multibody::ExternallyAppliedSpatialForce<double>>*
          fingers_control) const;

  void OutputBrickControl(
      const systems::Context<double>& context,
      std::vector<multibody::ExternallyAppliedSpatialForce<double>>*
          brick_control) const;

  int num_fingers_;
  int file_descriptor_{};
  multibody::BodyIndex brick_body_index_;
  // The mutex that guards buffer_;
  mutable std::mutex received_message_mutex_;

  std::vector<uint8_t> buffer_;

  int udp_message_size_;

  systems::OutputPortIndex qp_finger_control_output_port_{};
  systems::OutputPortIndex qp_brick_control_output_port_{};
  systems::AbstractStateIndex finger_control_state_index_{};
  systems::AbstractStateIndex brick_control_state_index_{};
};

// A system that subscribes to the QP planner and publishes to the QP planner.
class PlanarGripperQPControllerUDP : public systems::Diagram<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(PlanarGripperQPControllerUDP)
  PlanarGripperQPControllerUDP(int num_multibody_states,
                               multibody::BodyIndex brick_index,
                               int num_fingers, int num_brick_states,
                               int num_brick_accels, int local_port,
                               int remote_port, unsigned long remote_address,
                               double publish_period);
};
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
