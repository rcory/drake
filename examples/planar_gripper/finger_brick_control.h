#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/examples/planar_gripper/contact_force_qp.h"
#include "drake/examples/planar_gripper/planar_gripper.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/lcm/lcm_interface_system.h"

namespace drake {
namespace examples {
namespace planar_gripper {

using geometry::SceneGraph;
using lcm::DrakeLcm;
using multibody::ModelInstanceIndex;
using multibody::MultibodyPlant;
using systems::InputPort;
using systems::InputPortIndex;
using systems::OutputPort;
using systems::OutputPortIndex;

struct ForceControlOptions {
  double kpf_t_{0};  // Tangential proportional force gain (in brick frame)
  double kpf_n_{0};  // Normal proportional force gain (in brick frame)
  double kif_t_{0};  // Tangential integral force gain (in brick frame)
  double kif_n_{0};  // Normal integral force gain (in brick frame)
  double kp_t_{0};   // Tangential position gain (in brick frame)
  double kd_t_{0};   // Tangential derivative gain (in brick frame)
  double kp_n_{0};   // Normal position gain (in brick frame)
  double kd_n_{0};   // Normal derivative gain (in brick frame)
  Eigen::Matrix2d Kd_joint_{
      Eigen::Matrix2d::Zero()};  // joint damping (j1 & j2)
  double K_compliance_{0};       // impedance control stiffness
  double D_damping_{0};          // impedance control damping
  bool always_direct_force_control_{
      false};  // if false, do impedance control during non-contact
  Finger finger_to_control_{
      Finger::kFinger1};  // specifies which finger to control.
};

// Force controller with pure gravity compensation (no dynamics compensation
// yet).
// TODO(rcory) Should this class inherit from
//  systems::controllers::StateFeedbackControllerInterface?

/// A system that represents a force controller that can be in one of two
/// possible modes: 1) Direct-force control mode and 2) Impedance control mode.
///
/// The system input/output ports are described by the following:
/// @system{ ForceController,
///   @input_port{force_desired}
///   @input_port{finger_state_actual}
///   @input_port{plant_state_actual}
///   @input_port{tip_state_desired}
///   @input_port{contact_results}
///   @input_port{force_sensor_wrench}
///   @input_port{p_BrCb}
///   @input_port{is_in_contact}
///   @output_port{tau}
/// }
///
/// Case 1: Direct-force control mode (condition: finger is in contact).
///
/// Consider the robot dynamics
///   M(q)vdot + C(q,v)v = τ_g(q) + τ_external + τ_commanded,
/// where q == position, v == velocity, τ == torque, and τ_external represents
/// the generalized forces due to contact.
///
/// The direct-force controller produces τ_commanded as:
///  τ_commanded =  -τ_g(q) - (Kd * q) - (Jₜᵀ * f_command),
///  where f_command =
///         f_desired + (Kpf * Δf) + (Kpp * Δx) + (Kdp * Δẋ) + (Kif * ∫ Δf dt).
///
/// Jₜ: is the translational (only) jacobian of the contact point w.r.t. the
///     base finger frame (Ba).
/// f_desired: is a feedforward desired contact force.
/// Kp*: represent a positive definite gain matrix.
/// Δf = (f_desired - f_actual).
/// Δx = (x_desired - x_actual), where x is the contact point vector.
///
///
/// Case 2: Impedance control mode (condition: finger is NOT in contact).

class ForceController : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ForceController);
  ForceController(const MultibodyPlant<double>& plant,
                  const SceneGraph<double>& scene_graph,
                  ForceControlOptions options);

  ForceControlOptions get_options() const { return options_; }

  const InputPort<double>& get_force_desired_input_port() const {
    return this->get_input_port(force_desired_input_port_);
  }

  const InputPort<double>& get_finger_state_actual_input_port() const {
    return this->get_input_port(finger_state_actual_input_port_);
  }

  const InputPort<double>& get_plant_state_actual_input_port() const {
    return this->get_input_port(plant_state_actual_input_port_);
  }

  const InputPort<double>& get_contact_state_desired_input_port() const {
    return this->get_input_port(contact_state_desired_input_port_);
  }

  const InputPort<double>& get_force_sensor_input_port() const {
    return this->get_input_port(force_sensor_input_port_);
  }

  const InputPort<double>& get_geometry_query_input_port() const {
    return this->get_input_port(geometry_query_input_port_);
  }

  const systems::InputPort<double>& get_finger_face_assignments_input_port()
      const {
    return this->get_input_port(finger_face_assignments_input_port_);
  }

  const OutputPort<double>& get_torque_output_port() const {
    return this->get_output_port(torque_output_port_);
  }

  void CalcTauOutput(const systems::Context<double>& context,
                     systems::BasicVector<double>* output_vector) const;

 protected:
  void DoCalcTimeDerivatives(
      const systems::Context<double>& context,
      systems::ContinuousState<double>* derivatives) const override;

  void GetGains(EigenPtr<Matrix3<double>> Kp_force,
                EigenPtr<Matrix3<double>> Ki_force,
                EigenPtr<Matrix3<double>> Kp_position,
                EigenPtr<Matrix3<double>> Kd_position,
                const BrickFace& brick_face) const;

 private:
  const MultibodyPlant<double>& plant_;
  const SceneGraph<double>& scene_graph_;
  // This context is used solely for setting generalized positions and
  // velocities in plant_.
  std::unique_ptr<systems::Context<double>> plant_context_;
  InputPortIndex force_desired_input_port_{};
  InputPortIndex finger_state_actual_input_port_{};
  InputPortIndex plant_state_actual_input_port_{};
  InputPortIndex contact_state_desired_input_port_{};
  InputPortIndex force_sensor_input_port_{};
  InputPortIndex geometry_query_input_port_{};
  InputPortIndex finger_face_assignments_input_port_{};
  OutputPortIndex torque_output_port_{};
  ForceControlOptions options_;
};

/// Defines the type of control task. A "regulation" task controls the brick to
/// a set-point goal. A "tracking" task controls the brick to follow a desired
/// state trajectory.
enum class ControlTask {
  kRegulation,
  kTracking,
};

struct BrickGoal {
  double y_goal{0};  // goal y position (m).
  double z_goal{0};  // goal z position (m).
  double theta_goal{0};  // goal rotation angle (rad).
};

struct QPControlOptions {
  double T_{0};  // time horizon

  // The QP planner's timestep (enforced by a ZOH). This works in two ways:
  // 1) For a local QP controller, a zero order hold is placed at it's outputs.
  // 2) For a remote (LCM) controller, it is currently ignored.
  // TODO(rcory) Replace the LCM updates/publishes to use this parameter instead
  //  of kGripperLcmPeriod.
  double plan_dt{0.01};

  double QP_kp_r_{0};  // rotational proportional gain.
  double QP_kd_r_{0};  // rotational derivative gain.
  double QP_ki_r_{0};  // rotational integral gain.

  // translational proportional gains (kp_y, kp_z).
  Vector2d QP_Kp_t_{Vector2d::Zero()};
  // translational derivative gains (kd_y, kd_z).
  Vector2d QP_Kd_t_{Vector2d::Zero()};
  // translational integral gains (ki_y, ki_z).
  Vector2d QP_Ki_t_{Vector2d::Zero()};

  // rotation integral saturation limit.
  double QP_ki_r_sat_{0};
  // translation integral saturation limit (ki_y_sat, ki_z_sat).
  Vector2d QP_Ki_t_sat_{Vector2d::Zero()};

  double QP_weight_thetaddot_error_{0};  // thetaddot error weight.
  double QP_weight_acceleration_error_{0};  // tran. acceleration error weight.
  double QP_weight_f_Cb_B_{0};  // contact force magnitude penalty weight.
  double QP_mu_{0};  // coefficient of static friction between brick/fingertip.

  bool brick_only_{false};  // only control brick (no fingers).
  double viz_force_scale_{
      0};  // scale factor for visualizing spatial force arrow.

  // Brick specific parameters.
  double brick_rotational_damping_{0};     // brick's pin joint damping.
  double brick_translational_damping_{0};  // brick's translational damping.
  double brick_inertia_{0};                // brick's rotational inertia.
  double brick_mass_{0};                   // brick's mass.

  // Finger/face assignment map for *brick only* simulation.
  // Note: This map will *not* be used by the QP controller if `brick_only` is
  // false.
  std::unordered_map<Finger, BrickFaceInfo> brick_spatial_force_assignments_;

  BrickType brick_type_{BrickType::PinBrick};
  ControlTask control_task_{ControlTask::kRegulation};
  BrickGoal brick_goal_;  // for regulation task.
  trajectories::PiecewisePolynomial<double>
      desired_brick_traj_;  // for tracking task.
};

/// Connects a QP controller to a planar_gripper/brick simulation.
/// @param planar_gripper The planar gripper diagram.
/// @param lcm The LCM object used to connect signals to scope, as well as to
///        connect spatial forces to the visualizer.
/// @param finger_force_control_map An optional, that maps Fingers to
///        ForceController objects. For a brick only simulation, this optional
///        is nullopt.
/// @param qpoptions QP controller options structure.
/// @param builder A pointer to the diagram builder to which the QP controller
///        will be added.
void ConnectQPController(
    const PlanarGripper& planar_gripper, lcm::DrakeLcm* lcm,
    const std::optional<std::unordered_map<Finger, ForceController&>>&
        finger_force_control_map,
    const QPControlOptions qpoptions, systems::DiagramBuilder<double>* builder);

/// Connects an LCM QP controller to a planar_gripper/brick simulation.
/// @param planar_gripper The planar gripper diagram.
/// @param lcm The LCM object used to communicate with the remote LCM QP
///        controller. Also used to connect signals to scope, as well as to
///        connect spatial forces to the visualizer.
/// @param finger_force_control_map An optional, that maps Fingers to
///        ForceController objects. For a brick only simulation, this optional
///        is nullopt.
/// @param builder A pointer to the diagram builder to which the QP controller
///        will be added.
void ConnectLCMQPController(
    const PlanarGripper& planar_gripper, lcm::DrakeLcm* lcm,
    const std::optional<std::unordered_map<Finger, ForceController&>>&
        finger_force_control_map,
    const QPControlOptions& qpoptions,
    systems::DiagramBuilder<double>* builder);

/// Connects a UDP QP controller to a planar_gripper/brick simulation.
/// @param planar_gripper The planar gripper diagram.
/// @param lcm The LCM object used to connect signals to scope, as well as to
///        connect spatial forces to the visualizer.
/// @param finger_force_control_map An optional, that maps Fingers to
///        ForceController objects. For a brick only simulation, this optional
///        is nullopt.
/// @param publisher_local_port The local port of the UDP publisher for
///        publishing QP controller output.
/// @param publisher_remote_port The remote port for receiving the UDP message
///        published by the QP controller.
/// @param publisher_remote_address The unsiged long version of the IP address
///        for receiving the UDP message published by the QP controller.
/// @param receiver_local_port The local port for receiving the UDP message to
///        be used by the QP controller (like the plant state UDP message.)
/// @param builder A pointer to the diagram builder to which the QP controller
///        will be added.
void ConnectUDPQPController(
    const PlanarGripper& planar_gripper, lcm::DrakeLcm* lcm,
    const std::optional<std::unordered_map<Finger, ForceController&>>&
        finger_force_control_map,
    const QPControlOptions& qpoptions, int publisher_local_port,
    int publisher_remote_port, uint32_t publisher_remote_address,
    int receiver_local_port, systems::DiagramBuilder<double>* builder);

void AddGripperQPControllerToDiagram(
    const MultibodyPlant<double>& plant,
    systems::DiagramBuilder<double>* builder, const QPControlOptions& qpoptions,
    std::map<std::string, const InputPort<double>&>* in_ports,
    std::map<std::string, const OutputPort<double>&>* out_ports);

ForceController* SetupForceController(const PlanarGripper& planar_gripper,
                                      DrakeLcm* lcm,
                                      const ForceControlOptions& foptions,
                                      systems::DiagramBuilder<double>* builder);

/**
 *
 */

class PlantStateToFingerStateSelector : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(PlantStateToFingerStateSelector);
  PlantStateToFingerStateSelector(const MultibodyPlant<double>& plant,
                                  const Finger finger);

  void CalcOutput(const drake::systems::Context<double>& context,
                  drake::systems::BasicVector<double>* output) const;

 private:
  MatrixX<double> state_selector_matrix_;
};

/// This system has n vector-valued input ports of size `kNumJointsPerFinger`,
/// where n is the number of force controllers (typically one force controller
/// per finger). n is equal to the size of `finger_force_control_map`. Input
/// port n is a vector of actuation values for finger m, given as {fm_base_u,
/// fm_mid_u}, for the 2-joint case. The mapping of the nth input to the mth
/// finger is provided at construction, via `finger_force_control_map`. The
/// value n must be less than or equal to the number of fingers defined in the
/// MBP (and >= 1). If a force controller is not provided for any given finger
/// (i.e., n is less than the number of fingers defined in the MBP), then the
/// actuation input for those fingers will be set to zero.
// TODO(rcory) Allow gravity compensation (only) for fingers not associated with
//  a force controller.
///
/// This system has a single output port, which produces actuation
/// values for the entire gripper/brick MBP (in the plant's joint-actuator
/// ordering).
class ForceControllersToPlantActuationMap : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ForceControllersToPlantActuationMap);
  ForceControllersToPlantActuationMap(
      const MultibodyPlant<double>& plant,
      std::unordered_map<Finger, ForceController&> finger_force_control_map);

  void CalcOutput(const drake::systems::Context<double>& context,
                  drake::systems::BasicVector<double>* output) const;

  void ConnectForceControllersToPlant(
      const PlanarGripper& planar_gripper,
      systems::DiagramBuilder<double>* builder) const;

 private:
  MatrixX<double> actuation_selector_matrix_;
  const std::unordered_map<Finger, ForceController&> finger_force_control_map_;
  std::map<InputPortIndex, Finger> input_port_index_to_finger_map_;
};

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
