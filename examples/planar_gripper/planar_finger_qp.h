#pragma once

#include <memory>

#include "drake/examples/planar_gripper/gripper_brick.h"
#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"
#include "drake/systems/controllers/state_feedback_controller_interface.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace examples {
namespace planar_gripper {
/**
 * Solves a QP to find the contact force between the finger and the brick.
 *
 *     min w1 * |θ̈ - θ̈_des|² + w2 * |f_Cb_B|²
 *     s.t f_Cb_B in the friction cone.
 *
 * The desired angular acceleration is computed as θ̈_des = Kp * (θ_planned - θ)
 * + Kd * (θ̇_planned - θ̇) + θ̈ _planned
 */
class PlanarFingerInstantaneousQP {
 public:
  /**
   * @param plant The plant containing both the finger and the brick.
   * @param theta_planned The planned orientation of the brick.
   * @param thetadot_planned The planned angular velocity of the brick.
   * @param thetaddot_planned The planned angular acceleration of the brick.
   * @param Kp The proportional gain.
   * @param Kd The derivative gain.
   * @param theta The current brick orientation.
   * @param thetadot The current brick angular velocity.
   * @param p_BFingerTip The position of the finger tip bubble center measured
   * in the brick frame.
   * Note: This assumes that regardless of the position of the fingertip center,
   *  the fingertip surface is in contact with the brick and can apply a force at
   *  that point (in reality this may not be true).
   * @param weight_thetaddot_error The weight of the thetaddot error in the
   * cost.
   * @param weight_f_Cb The weight of the contact force in the cost.
   * @param contact_face The brick face that is in contact with the finger.
   * @param mu the friction coefficient between the finger tip and the brick.
   * @param I_B The inertia of the brick.
   * @param finger_tip_radius The radius of the bubble on the finger.
   * @param damping The damping at the brick's revolute joint.
   */
  PlanarFingerInstantaneousQP(
      const multibody::MultibodyPlant<double>* plant, double theta_planned,
      double thetadot_planned, double thetaddot_planned, double Kp, double Kd,
      double theta, double thetadot,
      const Eigen::Ref<const Eigen::Vector2d>& p_BFingerTip,
      double weight_thetaddot_error, double weight_f_Cb, BrickFace contact_face,
      double mu, double I_B, double finger_tip_radius, double damping);

  const solvers::MathematicalProgram& prog() const { return *prog_; }

  /** Returns the contact point Cb expressed in the brick frame. */
  const Eigen::Vector2d& p_BCb() const { return p_BCb_; }

  /**
   * Computes f_Cb_B, the contact force applied at the brick contact point Cb,
   * expressed in the brick frame.
   */
  const Eigen::Vector2d GetContactForceResult(
      const solvers::MathematicalProgramResult& result) const;

 private:
  const multibody::MultibodyPlant<double>* plant_;
  std::unique_ptr<solvers::MathematicalProgram> prog_;
  Eigen::Matrix2d friction_cone_edges_;
  Vector2<symbolic::Variable> f_Cb_B_edges_;
  Eigen::Vector2d p_BCb_;
};

class PlanarFingerInstantaneousQPController
    : public systems::controllers::StateFeedbackControllerInterface<double>,
      public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(PlanarFingerInstantaneousQPController)

  PlanarFingerInstantaneousQPController(
      const multibody::MultibodyPlant<double>* plant, double Kp, double Kd,
      double weight_thetaddot, double weight_f_Cb_B, double mu,
      double finger_tip_radius, double damping);

  const systems::InputPort<double>& get_input_port_estimated_state()
      const final {
    return this->get_input_port(input_index_state_);
  }

  /**
   * This port only takes the desired brick state, namely theta and thetadot.
   * Please do not connect the finger state to this port.
   */
  const systems::InputPort<double>& get_input_port_desired_state() const final {
    return this->get_input_port(input_index_desired_brick_state_);
  }

  const systems::InputPort<double>& get_input_port_desired_thetaddot() const {
    return this->get_input_port(input_index_desired_thetaddot_);
  }

  /**
   * This port takes in the current finger tip sphere center (y, z) position in
   * the brick frame. Notice that we ignore the x position since it is a planar
   * system.
   */
  const systems::InputPort<double>& get_input_port_p_BFingerTip() const {
    return this->get_input_port(input_index_p_BFingerTip_);
  }

  const systems::InputPort<double>& get_input_port_contact_face() const {
    return this->get_input_port(input_index_contact_face_);
  }

  const systems::OutputPort<double>& get_output_port_control() const final {
    return this->get_output_port(output_index_control_);
  }

 private:
  void CalcControl(
      const systems::Context<double>& context,
      std::vector<multibody::ExternallyAppliedSpatialForce<double>>* output)
      const;

  const multibody::MultibodyPlant<double>* plant_;
  double mu_;
  double I_B_;
  double Kp_;
  double Kd_;
  double weight_thetaddot_;
  double weight_f_Cb_B_;
  double finger_tip_radius_;
  double damping_;

  int brick_revolute_position_index_;
  multibody::BodyIndex brick_body_index_;

  int input_index_state_{-1};
  int input_index_desired_brick_state_{-1};
  int input_index_desired_thetaddot_{-1};
  int input_index_p_BFingerTip_{-1};
  int input_index_contact_face_{-1};
  int output_index_control_{-1};
};
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
