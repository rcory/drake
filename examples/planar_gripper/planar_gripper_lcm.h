#pragma once

/// @file
/// This file contains classes dealing with sending/receiving
/// LCM messages related to the planar gripper.
/// TODO(rcory) Create doxygen system diagrams for the classes below.
///
/// All (q, v) state vectors in this file are of the format
/// (joint_positions, joint_velocities).

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/examples/planar_gripper/planar_gripper_utils.h"
#include "drake/lcmt_planar_gripper_command.hpp"
#include "drake/lcmt_planar_gripper_finger_face_assignments.hpp"
#include "drake/lcmt_planar_gripper_status.hpp"
#include "drake/lcmt_planar_manipuland_desired.hpp"
#include "drake/lcmt_planar_manipuland_spatial_forces.hpp"
#include "drake/lcmt_planar_plant_state.hpp"
#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/event_status.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"

namespace drake {
namespace examples {
namespace planar_gripper {

using systems::InputPort;
using systems::OutputPort;

// This is rather arbitrary, for now.
// TODO(rcory) Refine this value once the planner comes online.
constexpr double kGripperLcmPeriod = 0.002;

/// Handles lcmt_planar_gripper_command messages from a LcmSubscriberSystem.
///
/// This system has one abstract valued input port which expects a
/// Value object templated on type `lcmt_planar_gripper_command`.
///
/// This system has two output ports. The first reports the commanded state
/// for all joints [q, v], and the second reports the commanded joint torques
/// (tau).
//
/// This system orders gripper joints according to q =
/// [q_finger_1, q_finger_2, ..., q_finger_n], where n is the number of fingers
/// and assumes each q_finger_i vector is ordered according to the preferred
/// finger joint ordering (see GetPreferredFingerJointOrdering in
/// planar_gripper_common.h). The same ordering applies to v and tau.
class GripperCommandDecoder : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GripperCommandDecoder)

  /// Constructor.
  /// @param num_fingers The total number of fingers used on the planar-gripper.
  explicit GripperCommandDecoder(int num_fingers = kNumFingers);

  /// Sets the initial position of the controlled gripper prior to any
  /// commands being received.  @p x contains the starting position.
  /// This position will be the commanded position (with zero
  /// velocity) until a position message is received.  If this
  /// function is not called, the starting position will be the zero
  /// configuration. The ordering of x follows the ordering of q described
  /// above.
  void set_initial_position(systems::Context<double>* context,
                            const Eigen::Ref<const VectorX<double>> x) const;

  const systems::OutputPort<double>& get_state_output_port() const {
    DRAKE_DEMAND(state_output_port_ != nullptr);
    return *state_output_port_;
  }

  const systems::OutputPort<double>& get_torques_output_port() const {
    DRAKE_DEMAND(torques_output_port_ != nullptr);
    return *torques_output_port_;
  }

 private:
  void OutputStateCommand(const systems::Context<double>& context,
                          systems::BasicVector<double>* output) const;

  void OutputTorqueCommand(const systems::Context<double>& context,
                           systems::BasicVector<double>* output) const;

  /// Event handler of the periodic discrete state update.
  systems::EventStatus UpdateDiscreteState(
      const systems::Context<double>& context,
      systems::DiscreteValues<double>* discrete_state) const;

  const int num_fingers_;
  const int num_joints_;
  const OutputPort<double>* state_output_port_{};
  const OutputPort<double>* torques_output_port_{};
};

/// Creates and outputs lcmt_planar_gripper_command messages.
///
/// This system has two vector-valued input ports containing the
/// desired state [q, v] on the first port, and commanded torque (tau) on the
/// second port.
///
/// This system assumes the gripper joints are ordered according to q =
/// [q_finger_1, q_finger_2, ..., q_finger_n], where n is the number of fingers
/// and each q_finger_i vector is ordered according to the preferred finger
/// joint ordering (see GetPreferredFingerJointOrdering in
/// planar_gripper_common.h). The same ordering applies to v and tau.
///
/// This system has one abstract valued output port that contains a
/// Value object templated on type `lcmt_planar_gripper_command`. Note
/// that this system does not actually send this message on an LCM
/// channel. To send the message, the output of this system should be
/// connected to an input port of a systems::lcm::LcmPublisherSystem
/// that accepts a Value object templated on type
/// `lcmt_planar_gripper_command`.
class GripperCommandEncoder : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GripperCommandEncoder)

  /// Constructor.
  /// @param num_joints The total number of fingers used on the planar-gripper.
  explicit GripperCommandEncoder(int num_fingers = kNumFingers);

  const systems::InputPort<double>& get_state_input_port() const {
    DRAKE_DEMAND(state_input_port_ != nullptr);
    return *state_input_port_;
  }

  const systems::InputPort<double>& get_torques_input_port() const {
    DRAKE_DEMAND(torques_input_port_ != nullptr);
    return *torques_input_port_;
  }

 private:
  void OutputCommand(const systems::Context<double>& context,
                     lcmt_planar_gripper_command* output) const;

  const int num_fingers_;
  const int num_joints_;
  const InputPort<double>* state_input_port_{};
  const InputPort<double>* torques_input_port_{};
};

/// Handles lcmt_planar_gripper_status messages from a LcmSubscriberSystem.
///
/// This system has one abstract valued input port which expects a
/// Value object templated on type `lcmt_planar_gripper_status`.
///
/// This system has two vector valued output ports which report
/// measured position and velocity (state) as well as fingertip forces (fy, fz).
///
/// All ports will continue to output their initial state (typically
/// zero) until a message is received.
class GripperStatusDecoder : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GripperStatusDecoder)

  /// Constructor.
  /// @param num_fingers The total number of fingers used on the planar-gripper.
  explicit GripperStatusDecoder(int num_fingers = kNumFingers);

  const systems::OutputPort<double>& get_state_output_port() const {
    DRAKE_DEMAND(state_output_port_ != nullptr);
    return *state_output_port_;
  }

  const systems::OutputPort<double>& get_force_output_port() const {
    DRAKE_DEMAND(force_output_port_ != nullptr);
    return *force_output_port_;
  }

 private:
  void OutputStateStatus(const systems::Context<double>& context,
                         systems::BasicVector<double>* output) const;

  void OutputForceStatus(const systems::Context<double>& context,
                         systems::BasicVector<double>* output) const;

  /// Event handler of the periodic discrete state update.
  systems::EventStatus UpdateDiscreteState(
      const systems::Context<double>& context,
      systems::DiscreteValues<double>* discrete_state) const;

  const int num_fingers_;
  const int num_joints_;
  const int num_tip_forces_;
  const OutputPort<double>* state_output_port_{};
  const OutputPort<double>* force_output_port_{};
};

/// Creates and outputs lcmt_planar_gripper_status messages.
///
/// This system has two vector-valued input ports containing the
/// current position and velocity (state) as well as fingertip forces (fy, fz).
///
/// This system has one abstract valued output port that contains a
/// Value object templated on type `lcmt_planar_gripper_status`. Note that this
/// system does not actually send this message on an LCM channel. To send the
/// message, the output of this system should be connected to an input port of
/// a systems::lcm::LcmPublisherSystem that accepts a
/// Value object templated on type `lcmt_planar_gripper_status`.
class GripperStatusEncoder : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GripperStatusEncoder)

  /// Constructor.
  /// @param num_joints The total number of fingers used on the planar-gripper.
  explicit GripperStatusEncoder(int num_fingers = kNumFingers);

  const systems::InputPort<double>& get_state_input_port() const {
    DRAKE_DEMAND(state_input_port_ != nullptr);
    return *state_input_port_;
  }

  const systems::InputPort<double>& get_force_input_port() const {
    DRAKE_DEMAND(force_input_port_ != nullptr);
    return *force_input_port_;
  }

 private:
  // This is the method to use for the output port allocator.
  lcmt_planar_gripper_status MakeOutputStatus() const;

  // This is the calculator method for the output port.
  void OutputStatus(const systems::Context<double>& context,
                    lcmt_planar_gripper_status* output) const;

  const int num_fingers_;
  const int num_joints_;
  const int num_tip_forces_;
  const InputPort<double>* state_input_port_{};
  const InputPort<double>* force_input_port_{};
};

/// =================== QP Controller Section ===========================

// TODO(rcory) Make the "decoder" classes take in an LCM period as an argument.
//  Currently, it the discrete variable update is hard-coded to be
//  kGripperLCMPeriod.

/*
 * This takes in an lcmt_planar_manipuland_spatial_forces object and outputs
 * a std::vector<multibody::ExternallyAppliedSpatialForce<double>> object.
 */
class QPBrickControlDecoder : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QPBrickControlDecoder)

  explicit QPBrickControlDecoder(multibody::BodyIndex brick_body_index);

 private:
  void OutputBrickControl(
      const systems::Context<double>& context,
      std::vector<multibody::ExternallyAppliedSpatialForce<double>>*
          brick_control) const;

  /// Event handler of the periodic discrete state update.
  systems::EventStatus UpdateAbstractState(
      const systems::Context<double>& context,
      systems::State<double>* state) const;

  multibody::BodyIndex brick_body_index_;
};

/*
 * This takes in a
 * std::vector<multibody::ExternallyAppliedSpatialForce<double>> object and
 * outputs a lcmt_planar_manipuland_spatial_forces object.
 */
class QPBrickControlEncoder : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QPBrickControlEncoder)

  QPBrickControlEncoder();

 private:
  void EncodeBrickControl(
      const systems::Context<double>& context,
      lcmt_planar_manipuland_spatial_forces* spatial_forces_lcm) const;
};

/*
 * This takes in an lcmt_planar_manipuland_spatial_forces object and outputs a
 * std::unordered_map<Finger, multibody::ExternallyAppliedSpatialForce<double>>.
 * The incoming lcmt object must contain exactly `num_fingers` spatial forces.
 */
class QPFingersControlDecoder : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QPFingersControlDecoder)

  explicit QPFingersControlDecoder(multibody::BodyIndex brick_body_index);

 private:
  void OutputFingersControl(
      const systems::Context<double>& context,
      std::unordered_map<Finger,
                         multibody::ExternallyAppliedSpatialForce<double>>*
          fingers_control) const;

  /// Event handler of the periodic discrete state update.
  systems::EventStatus UpdateAbstractState(
      const systems::Context<double>& context,
      systems::State<double>* state) const;

  const multibody::BodyIndex brick_body_index_;
};

/*
 * This class takes in a
 * std::unordered_map<Finger, multibody::ExternallyAppliedSpatialForce<double>>
 * and outputs an lcmt_planar_manipuland_spatial_forces object.
 */
class QPFingersControlEncoder : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QPFingersControlEncoder)

  QPFingersControlEncoder();

 private:
  void EncodeFingersControl(
      const systems::Context<double>& context,
      lcmt_planar_manipuland_spatial_forces* spatial_forces_lcm) const;
};

/*
 * This class takes in an lcmt_planar_plant_state and outputs
 * a VectorX<double> of the same size as the lcmt's `plant_state`. The ordering
 * is assumed to match MBP joint ordering.
 */
class QPEstimatedStateDecoder : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QPEstimatedStateDecoder)

  explicit QPEstimatedStateDecoder(const int num_plant_states);

 private:
  void OutputEstimatedState(const systems::Context<double>& context,
                            systems::BasicVector<double>* plant_state) const;

  /// Event handler of the periodic discrete state update.
  systems::EventStatus UpdateDiscreteState(
      const systems::Context<double>& context,
      systems::DiscreteValues<double>* discrete_state) const;

  const int num_plant_states_;
};

/*
 * This class takes in a VectorX<double> (plant state) and outputs an
 * lcmt_planar_plant_state object. The ordering is assumed to match MBP joint
 * ordering.
 */
class QPEstimatedStateEncoder : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QPEstimatedStateEncoder)

  explicit QPEstimatedStateEncoder(const int num_plant_states);

 private:
  void EncodeEstimatedState(const systems::Context<double>& context,
                            lcmt_planar_plant_state* plant_status) const;

  const int num_plant_states_;
};

/*
 * This class takes in an lcmt_planar_gripper_finger_face_assignments object
 * and outputs a std::unordered_map<Finger, BrickFaceInfo> object.
 */
class QPFingerFaceAssignmentsDecoder : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QPFingerFaceAssignmentsDecoder)

  QPFingerFaceAssignmentsDecoder();

 private:
  void OutputFingerFaceAssignments(
      const systems::Context<double>& context,
      std::unordered_map<Finger, BrickFaceInfo>* finger_face_assignments) const;

  /// Event handler of the periodic discrete state update.
  systems::EventStatus UpdateAbstractState(
      const systems::Context<double>& context,
      systems::State<double>* state) const;
};

/*
 * This class takes in a
 * std::unordered_map<Finger, BrickFaceInfo> object and outputs an
 * lcmt_planar_gripper_finger_face_assignments object. Note that an
 * lcmt_planar_gripper_finger_face_assignments object contains a
 * `finger_face_assignment` array, whose entries are identified by the
 * `finger_name` field, i.e., the array ordering should not be assumed to be in
 * any particular order.
 */
class QPFingerFaceAssignmentsEncoder : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QPFingerFaceAssignmentsEncoder)

  QPFingerFaceAssignmentsEncoder();

 private:
  void EncodeFingerFaceAssignments(const systems::Context<double>& context,
                                   lcmt_planar_gripper_finger_face_assignments*
                                       finger_face_assignments_lcm) const;
};

/*
 * This class takes in a lcmt_planar_gripper_qp_brick_desired object and outputs
 * (1) a VectorXd of desired brick state and (2) a VectorXd of desired brick
 * accelerations.
 */
class QPBrickDesiredDecoder : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QPBrickDesiredDecoder)

  QPBrickDesiredDecoder(int num_brick_states, int num_brick_accels);

 private:
  void OutputBrickDesiredState(
      const systems::Context<double>& context,
      systems::BasicVector<double>* qp_desired_brick_state) const;

  void DecodeBrickDesiredAccel(
      const systems::Context<double>& context,
      systems::BasicVector<double>* qp_desired_brick_accels) const;

  /// Event handler of the periodic discrete state update.
  systems::EventStatus UpdateDiscreteState(
      const systems::Context<double>& context,
      systems::DiscreteValues<double>* discrete_state) const;

  const int num_brick_states_;
  const int num_brick_accels_;

  systems::DiscreteStateIndex brick_state_index_{};
  systems::DiscreteStateIndex brick_accel_index_{};
};

/*
 * This class takes in two inputs 1) a VectorXd of desired brick state and 2)
 * a VectorXd of desired brick accelerations, and outputs an
 * lcmt_planar_gripper_qp_brick_desired object.
 */
class QPBrickDesiredEncoder : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QPBrickDesiredEncoder)

  QPBrickDesiredEncoder(int num_brick_states, int num_brick_accels);

 private:
  void EncodeBrickDesired(
      const systems::Context<double>& context,
      lcmt_planar_manipuland_desired* planar_gripper_qp_brick_desired) const;

  const int num_brick_states_;
  const int num_brick_accels_;
};

// A system that subscribes to the QP planer and publishes to the QP planer.
class PlanarGripperQPControllerLCM : public systems::Diagram<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(PlanarGripperQPControllerLCM)
  PlanarGripperQPControllerLCM(const int num_multibody_states,
                               const int num_brick_states,
                               const int num_brick_accels,
                               const multibody::BodyIndex brick_index,
                               drake::lcm::DrakeLcmInterface* lcm,
                               double publish_period);
};

// A system that subscribes to the simulation and publishes to the simulation.
class PlanarGripperSimulationLCM : public systems::Diagram<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(PlanarGripperSimulationLCM)
  PlanarGripperSimulationLCM(int num_multibody_states, int num_brick_states,
                             int num_brick_accels,
                             drake::lcm::DrakeLcmInterface* lcm,
                             double publish_period);

  const systems::lcm::LcmSubscriberSystem& get_estimated_plant_state_sub() {
    return *qp_estimated_plant_state_sub_;
  }
  const systems::lcm::LcmSubscriberSystem& get_finger_face_assignments_sub() {
    return *qp_finger_face_assignments_sub_;
  }
  const systems::lcm::LcmSubscriberSystem& get_brick_desired_sub() {
    return *qp_brick_desired_sub_;
  }

 private:
  systems::lcm::LcmSubscriberSystem* qp_estimated_plant_state_sub_;
  systems::lcm::LcmSubscriberSystem* qp_finger_face_assignments_sub_;
  systems::lcm::LcmSubscriberSystem* qp_brick_desired_sub_;
};

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
