#pragma once

#include "drake/multibody/plant/multibody_plant.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/examples/planar_gripper/planar_gripper.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/examples/planar_gripper/contact_force_qp.h"

namespace drake {
namespace examples {
namespace planar_gripper {

using lcm::DrakeLcm;
using geometry::SceneGraph;
using multibody::MultibodyPlant;
using multibody::ModelInstanceIndex;
using systems::InputPortIndex;
using systems::InputPort;
using systems::OutputPortIndex;
using systems::OutputPort;

struct ForceControlOptions{
  double kpf_t_{0};  // Tangential proportional force gain (in brick frame)
  double kpf_n_{0};  // Normal proportional force gain (in brick frame)
  double kif_t_{0};  // Tangential integral force gain (in brick frame)
  double kif_n_{0};  // Normal integral force gain (in brick frame)
  double kp_t_{0};  // Tangential position gain (in brick frame)
  double kd_t_{0};  // Tangential derivative gain (in brick frame)
  double kp_n_{0};  // Normal position gain (in brick frame)
  double kd_n_{0};  // Normal derivative gain (in brick frame)
  Eigen::Matrix2d Kd_joint_{
      Eigen::Matrix2d::Zero()};  // joint damping (j1 & j2)
  double K_compliance_{0};  // impedance control stiffness
  double D_damping_{0};  // impedance control damping
  double brick_damping_{0};  // brick pin joint damping
  double brick_inertia_{0};  // brick's rotational inertia
  bool always_direct_force_control_{true};  // false for impedance control during non-contact
  Finger finger_to_control_{Finger::kFinger1};  // specifies which finger to control.

  // TODO(rcory) Compute this on the fly instead.
  BrickFace brick_face_;
};

// Force controller with pure gravity compensation (no dynamics compensation
// yet).
// TODO(rcory) Should this class inherit from
//  systems::controllers::StateFeedbackControllerInterface?
class ForceController : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ForceController);
  ForceController(const MultibodyPlant<double>& plant,
                  const SceneGraph<double>& scene_graph,
                  ForceControlOptions options,
                  const ModelInstanceIndex gripper_index,
                  const ModelInstanceIndex brick_index);

  ForceControlOptions get_options() const {
    return options_;
  }

  const InputPort<double>& get_force_desired_input_port() const {
    return this->get_input_port(force_desired_input_port_);
  }

  const InputPort<double>& get_finger_state_actual_input_port() const {
    return this->get_input_port(finger_state_actual_input_port_);
  }

  const InputPort<double>& get_brick_state_actual_input_port() const {
    return this->get_input_port(brick_state_actual_input_port_);
  }

  const InputPort<double>& get_tip_state_desired_input_port() const {
    return this->get_input_port(tip_state_desired_input_port_);
  }

  const InputPort<double>& get_contact_results_input_port() const {
    return this->get_input_port(contact_results_input_port_);
  }

  const InputPort<double>& get_force_sensor_input_port() const {
    return this->get_input_port(force_sensor_input_port_);
  }

  const InputPort<double>& get_accelerations_actual_input_port() const {
    return this->get_input_port(accelerations_actual_input_port_);
  }

  const InputPort<double>& get_geometry_query_input_port() const {
    return this->get_input_port(geometry_query_input_port_);
  }

  const InputPort<double>& get_contact_point_ref_accel_input_port() const {
    return this->get_input_port(contact_point_ref_accel_input_port_);
  }

  const InputPort<double>& get_is_contact_input_port() const {
    return this->get_input_port(is_contact_input_port_);
  }

  /**
   * This port takes in the current finger tip sphere center (y, z) position in
   * the brick frame. Notice that we ignore the x position since it is a planar
   * system.
   */
  const systems::InputPort<double>& get_p_BrFingerTip_input_port() const {
    return this->get_input_port(p_BrFingerTip_input_port_);
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
                EigenPtr<Matrix3<double>> Kd_position) const;

 private:
  const MultibodyPlant<double>& plant_;
  const SceneGraph<double>& scene_graph_;
  // This context is used solely for setting generalized positions and
  // velocities in plant_.
  std::unique_ptr<systems::Context<double>> plant_context_;
  InputPortIndex force_desired_input_port_{};
  InputPortIndex finger_state_actual_input_port_{};
  InputPortIndex gripper_state_actual_input_port_{};
  InputPortIndex brick_state_actual_input_port_{};
  InputPortIndex tip_state_desired_input_port_{};
  InputPortIndex contact_results_input_port_{};
  InputPortIndex force_sensor_input_port_{};
  InputPortIndex accelerations_actual_input_port_{};
  InputPortIndex geometry_query_input_port_{};
  InputPortIndex contact_point_ref_accel_input_port_{};
  InputPortIndex p_BrFingerTip_input_port_{};
  InputPortIndex is_contact_input_port_{};
  OutputPortIndex torque_output_port_{};
  ModelInstanceIndex gripper_index_{};
  ModelInstanceIndex brick_index_{};
  ForceControlOptions options_;
};

struct QPControlOptions{
  double T_{0};  // time horizon
  double theta0_{0};  // initial rotation angle (rad)
  double thetaf_{0};  // final rotation angle (rad)

  double QP_Kp_{0};  // QP controller Kp gain
  double QP_Kd_{0};  // QP controller Kd gain
  double QP_weight_thetaddot_error_{0};  // thetaddot error weight
  double QP_weight_f_Cb_B_{0};  // contact force magnitude penalty weight
  double QP_mu_{0};  // QP mu value

  bool brick_only_{false};  // only control brick (no finger)
  double viz_force_scale_{0};  // scale factor for visualizing spatial force arrow.

  // Brick specific parameters.
  double brick_damping_{0};  // brick's pin joint damping.
  double brick_inertia_{0};  // brick's rotational inertia.

  // The brick's finger/contact-face assignments.
  std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>
      finger_face_assignments_;

  BrickType brick_type_{BrickType::PinBrick};
};

void ConnectQPController(
    const PlanarGripper& planar_gripper, lcm::DrakeLcm& lcm,
    const std::optional<std::unordered_map<Finger, ForceController&>>&
        finger_force_control_map,
    const QPControlOptions qpoptions, systems::DiagramBuilder<double>* builder);

ForceController* SetupForceController(const PlanarGripper& planar_gripper,
                                      DrakeLcm& lcm,
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

/// Has two input ports. The first input is a vector of actuation values
/// for the planar-gripper (6 x 1, in joint-actuator ordering). Typically this
/// is the output of GeneralizedForceToActuationOrdering(control_plant). The
/// second input is a vector of actuation values for finger n given as:
/// {fn_base_u, fn_mid_u}.
/// This system has a single output port, which produces actuation
/// values for the gripper/brick MBP (in the plant's joint-actuator ordering).
class FingerToPlantActuationMap : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FingerToPlantActuationMap);
  FingerToPlantActuationMap(const MultibodyPlant<double>& plant,
                             const Finger finger);

  void CalcOutput(const drake::systems::Context<double>& context,
                  drake::systems::BasicVector<double>* output) const;

 private:
  MatrixX<double> actuation_selector_matrix_;
  MatrixX<double> actuation_selector_matrix_inv_;
  const Finger finger_;
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