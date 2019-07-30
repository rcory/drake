#pragma once

#include "drake/common/drake_copyable.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/controllers/admittance_controller.h"
#include "drake/systems/controllers/pid_controller.h"
#include "drake/systems/controllers/state_feedback_controller_interface.h"
#include "drake/systems/framework/diagram.h"

namespace drake {
namespace systems {
namespace controllers {

using multibody::MultibodyPlant;
using systems::InputPortIndex;
using systems::OutputPortIndex;

/**
 * Implements an admittance controller that regulates a desired force at the
 * end-effector. Admittance control means we regulate the robot motion (output)
 * as a result of the contact force (input). The controller is designed to
 * regulate robot joint velocities as a function of contact-force error. For a
 * position controlled robot, desired joint velocities are integrated to
 * produce desired joint positions trajectories.
 */
template <typename T>
class AdmittanceController : public Diagram<T>,
                                  public StateFeedbackControllerInterface<T> {

 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(AdmittanceController)

  AdmittanceController(const MultibodyPlant<T> plant);

  ~AdmittanceController();

  /**
   * Returns the input port for the estimated state.
   */
  const InputPort<T>& get_input_port_estimated_state() const final {
    return this->get_input_port(input_port_index_estimated_state_);
  }

  /**
   * Returns the input port for the desired state.
   */
  const InputPort<T>& get_input_port_desired_state() const final {
    return this->get_input_port(input_port_index_desired_state_);
  }

  /**
   * Returns the output port for computed control.
   */
  const OutputPort<T>& get_output_port_control() const final {
    return this->get_output_port(output_port_index_control_);
  }

 private:
  const multibody::MultibodyPlant<T>& plant_;
  PidController<T>* pid_{nullptr};
  InputPortIndex input_port_index_estimated_state_{-1};
  InputPortIndex input_port_index_desired_state_{-1};
  OutputPortIndex output_port_index_control_{-1};
};

}  // namespace drake
}  // namespace systems
}  // namespace drake