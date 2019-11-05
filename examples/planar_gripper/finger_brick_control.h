#pragma once

#include "drake/multibody/plant/multibody_plant.h"
#include "drake/lcm/drake_lcm.h"

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
  double kfy_{0};  // y-axis force gain (in brick frame)
  double kfz_{0};  // z-axis force gain (in brick frame)
  double kpz_{0};  // z-axis position gain (in brick frame)
  double kdz_{0};  // z-axis derivative gain (in brick frame)
  double Kd_{0};  // joint damping
  double K_compliance_{0};  // impedance control stiffness
  double D_damping_{0};  // impedance control damping
  bool always_direct_force_control_{true};  // false for impedance control during non-contact
};

// Force controller with pure gravity compensation (no dynamics compensation
// yet). Regulates position in z, and force in y.
class ForceController : public systems::LeafSystem<double> {
 public:
  ForceController(MultibodyPlant<double>& plant,
                  SceneGraph<double>& scene_graph, ForceControlOptions options);

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

  const OutputPort<double>& get_torque_output_port() const {
    return this->get_output_port(torque_output_port_);
  }

  void CalcTauOutput(const systems::Context<double>& context,
                     systems::BasicVector<double>* output_vector) const;

 private:
  MultibodyPlant<double>& plant_;
  SceneGraph<double>& scene_graph_;
  // This context is used solely for setting generalized positions and
  // velocities in plant_.
  std::unique_ptr<systems::Context<double>> plant_context_;
  InputPortIndex force_desired_input_port_{};
  InputPortIndex finger_state_actual_input_port_{};
  InputPortIndex brick_state_actual_input_port_{};
  InputPortIndex tip_state_desired_input_port_{};
  InputPortIndex contact_results_input_port_{};
  OutputPortIndex torque_output_port_{};
  ModelInstanceIndex finger_index_{};
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

  // sets the brick damping to zero, instead of taking the .sdf value.
  bool assume_zero_brick_damping_{false};

  // These two (yc, zc) only affect the brick-only simulation.
  double yc_{0};  // y contact point, for brick only QP
  double zc_{0};  // z contact point, for brick only QP
};

/// A method that connects the finger/brick QP controller to the force
/// controller.
void ConnectControllers(const MultibodyPlant<double>& plant,
                        lcm::DrakeLcm& lcm,
                        const ForceController& force_controller,
                        const ModelInstanceIndex& brick_index,
                        const QPControlOptions options,
                        systems::DiagramBuilder<double>* builder);

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake