#pragma once

#include "drake/multibody/plant/multibody_plant.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/examples/planar_gripper/planar_gripper.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"

namespace drake {
namespace examples {
namespace planar_gripper {

using geometry::SceneGraph;
using multibody::MultibodyPlant;
using multibody::ModelInstanceIndex;
using systems::InputPortIndex;
using systems::InputPort;
using systems::OutputPortIndex;
using systems::OutputPort;

struct ForceControlOptions{
  double kpfy_{0};  // y-axis proportional force gain (in brick frame)
  double kpfz_{0};  // z-axis proportional force gain (in brick frame)
  double kify_{0};  // y-axis integral force gain (in brick frame)
  double kifz_{0};  // z-axis integral force gain (in brick frame)
  double kpy_{0};  // y-axis position gain (in brick frame)
  double kdy_{0};  // y-axis derivative gain (in brick frame)
  double kpz_{0};  // z-axis position gain (in brick frame)
  double kdz_{0};  // z-axis derivative gain (in brick frame)
  Eigen::Matrix2d Kd_{Eigen::Matrix2d::Zero()};  // joint damping (j1 & j2)
  double K_compliance_{0};  // impedance control stiffness
  double D_damping_{0};  // impedance control damping
  double brick_damping_{0};  // brick pin joint damping
  double brick_inertia_{0};  // brick's rotational inertia
  bool always_direct_force_control_{true};  // false for impedance control during non-contact
  Finger finger_to_control_{Finger::kFinger1};  // specifies which finger to control.
};

// Force controller with pure gravity compensation (no dynamics compensation
// yet).
// TODO(rcory) Should this class inherit from
//  systems::controllers::StateFeedbackControllerInterface?
class ForceController : public systems::LeafSystem<double> {
 public:
  ForceController(const MultibodyPlant<double>& plant,
                  const SceneGraph<double>& scene_graph,
                  ForceControlOptions options, ModelInstanceIndex gripper_index,
                  ModelInstanceIndex brick_index);

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

  // These two (yc, zc) only affect the brick-only simulation.
  double yc_{0};  // y contact point, for brick only QP
  double zc_{0};  // z contact point, for brick only QP

  // Brick specific parameters.
  double brick_damping_{0};  // brick's pin joint damping.
  double brick_inertia_{0};  // brick's rotational inertia.

  // The brick's contact face.
  BrickFace contact_face_{BrickFace::kPosZ};
};

/// A method that connects the finger/brick QP controller to the force
/// controller.
void ConnectQPController(const MultibodyPlant<double>& plant,
                         const geometry::SceneGraph<double>& scene_graph,
                         lcm::DrakeLcm& lcm,
                         const ForceController& force_controller,
                         const ModelInstanceIndex& brick_index,
                         const QPControlOptions options,
                         systems::DiagramBuilder<double>* builder);

void ConnectQPController(PlanarGripper& planar_gripper, lcm::DrakeLcm& lcm,
                         const ForceController& force_controller,
                         const QPControlOptions qpoptions,
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
/// for the planar-gripper (6 x 1, in joint actuator ordering). Typically this
/// is the output of GeneralizedForceToActuationOrdering(control_plant). The
/// second input is a vector of actuation values for finger 3 given as:
/// {f3_base_u, f3_mid_u}.
// TODO(rcory) I should generalize this beyond just finger 3.
/// This system has a single output port, which produces actuation
/// values for the gripper/brick MBP (in the plant's actuator ordering).
class FingersToPlantActuationMap : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FingersToPlantActuationMap);
  FingersToPlantActuationMap(const MultibodyPlant<double>& plant,
                             const Finger finger);

  void CalcOutput(const drake::systems::Context<double>& context,
                  drake::systems::BasicVector<double>* output) const;

 private:
  MatrixX<double> actuation_selector_matrix_;
  MatrixX<double> actuation_selector_matrix_inv_;
  const Finger finger_;
};

/// Creates the QP controller (finger/brick for now), and connects it to the
/// force controller (this is for the lcm based finger/brick rotate).
// TODO(rcory) Reconcile this with ConnectControllers in finger_brick_control.cc
void ConnectAllControllers(PlanarGripper& planar_gripper,
                           lcm::DrakeLcm& lcm,
                           const ForceController& force_controller,
                           const QPControlOptions qpoptions,
                           systems::DiagramBuilder<double>* builder);

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake