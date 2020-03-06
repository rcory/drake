#pragma once

#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"
#include "drake/systems/controllers/state_feedback_controller_interface.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"

namespace drake {
namespace examples {
namespace planar_gripper {

enum class BrickType {
  PinBrick,
  PlanarBrick,
};

/**
 * Solves a QP to find the contact force from the finger to the object, such
 * that the object is steered back to the desired trajectory. Note this only
 * finds the contact force in the next dt.
 * Mathematically, we solve the following quadratic programming problem
 *
 *     min   w₁ * |p̈_WB - p̈ᵈᵉˢ_WB|² + w₂  * |θ̈ - θ̈ᵈᵉˢ|² + w₃ * |f_Cb_B|²
 *     s.t   m * p̈_WB = m * g_W + R_WB * ∑ f_Cb_B
 *           I * θ̈ = ∑ p_Cb_B * f_Cb_B
 *           f_Cb_B in the friction cone.
 * where p̈ᵈᵉˢ_WB = Kp1 * (p_planned_WB - p_WB) + Kd1 * (ṗ_planned_WB - ṗ_WB) +
 * p̈_planned_WB, and θ̈ᵈᵉˢ = Kp2 * (θ_planned - θ) + Kd2 * (θ̇_planned - θ̇ ) + θ̈
 * _planned.
 */
class InstantaneousContactForceQP {
 public:
  /**
   * @param gripper_brick The helper class. Must remain active during the
   * lifetime of this QP.
   * @param p_WB_planned The planned position of the brick in the world frame.
   * @param v_WB_planned The planned velocity of the brick in the world frame.
   * @param a_WB_planned The planned acceleration of the brick in the worl
   * frame.
   * @param theta_planned The planned orientation of the brick.
   * @param thetadot_planned The planned angular velocity of the brick.
   * @param thetaddot_planned The planned angular acceleration of the brick.
   * @param Kp_t The proportional gain of the brick translational position.
   * @param Kd_t The derivative gain of the brick translational velocity.
   * @param Kp_r The proportional gain of the brick rotational angle.
   * @param Kd_r The derivative gain of the brick rotational (angular) velocity.
   * @param brick_state The brick state include [p_y, p_z, θ, ṗ_y, ṗ_z, θ̇],
   * namely the brick position/orientation and its velocities.
   * @param finger_face_assignments a std::unordered_map which assigns a Finger
   * to a contact face on the brick (BrickFace) and to a contact point on the
   * brick. finger_face_assignments[Finger] gives a std::pair(BrickFace,
   * p_BoBq_B), where p_BoBq_B is the y/z location of the contact point Bq,
   * measured from the brick's body origin Bo, expressed in the brick's body
   * frame B.
   * @param weight_a_error The weight of the acceleration error in the cost.
   * This error is made up of a PD term and a FF (planned) term.
   * @param weight_thetaddot_error The weight of the angular acceleration error
   * in the cost. This error is made up of a PD term and a FF (planned) term.
   * @param weight_f_Cb The weight of the contact force in the cost.
   * @param mu The brick/floor coefficient of static friction (stiction).
   * @param I_B The brick's rotational moment of inertia around its axis of
   * rotation.
   * @param mass_B The mass of the brick.
   * @param rotational_damping The rotational viscous damping coefficient.
   * @param translational_damping The translational viscous damping coefficient.
   */
  InstantaneousContactForceQP(
      const BrickType brick_type,
      const Eigen::Ref<const Eigen::Vector2d>& p_WB_planned,
      const Eigen::Ref<const Eigen::Vector2d>& v_WB_planned,
      const Eigen::Ref<const Eigen::Vector2d>& a_WB_planned,
      double theta_planned, double thetadot_planned, double thetaddot_planned,
      const Eigen::Ref<const Eigen::Matrix2d>& Kp_t,
      const Eigen::Ref<const Eigen::Matrix2d>& Kd_t, double Kp_r, double Kd_r,
      const Eigen::Ref<const Vector6<double>>& brick_state,
      const std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>&
          finger_face_assignments,
      double weight_a_error, double weight_thetaddot_error, double weight_f_Cb,
      double mu, double I_B, double mass_B, double rotational_damping,
      double translational_damping);

  const solvers::MathematicalProgram& prog() const { return *prog_; }

  /**
   * Returns (f_Cb_B, p_Cb_B) f_Cb_B is the contact force applied by the finger
   * on the brick, expressed in the brick frame. p_Cb_B is the contact point Cb
   * expressed in the brick frame.
   */
  std::unordered_map<Finger, std::pair<Eigen::Vector2d, Eigen::Vector2d>>
  GetContactForceResult(
      const solvers::MathematicalProgramResult& result) const;

 private:
  struct FingerFaceContact {
    FingerFaceContact(Finger m_finger, BrickFace m_brick_face,
                      const Vector2<symbolic::Variable>& m_f_Cb_B_edges,
                      double m_mu, const Eigen::Vector2d& m_p_BCb)
        : finger{m_finger},
          brick_face{m_brick_face},
          f_Cb_B_edges{m_f_Cb_B_edges},
          mu{m_mu},
          p_BCb_{m_p_BCb} {}
    Finger finger;
    BrickFace brick_face;
    Vector2<symbolic::Variable> f_Cb_B_edges;
    double mu;
    Vector2<double> p_BCb_;
  };
  std::unique_ptr<solvers::MathematicalProgram> prog_;
  std::vector<FingerFaceContact> finger_face_contacts_;
};

class InstantaneousContactForceQPController
    : public systems::controllers::StateFeedbackControllerInterface<double>,
      public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(InstantaneousContactForceQPController)

  /**
   * @param gripper_brick The gripper_brick system.
   * @param Kp_tr The proportional gain for the brick y/z translational
   * position.
   * @param Kd_tr The derivative gain for the brick y/z translational position.
   * @param Kp_ro The proportional gain for the brick rotational position
   * (angle).
   * @param Kd_ro The derivative gain for the brick rotational velocity.
   * @param weight_a_error The weighting for the brick y/z acceleration in the
   * cost. This error is made up of a PD term and a FF (planned) term.
   * @param weight_thetaddot_error The weighting for the brick thetaddot in the
   * cost. This error is made up of a PD term and a FF (planned) term.
   * @param weight_f_Cb_B The weighting for the contact forces in the cost.
   * @param mu The friction coefficient used to compute friction cones.
   *        Note: This value is strictly used for modeling the frictional force
   *        between the fingertip and the brick, and NOT between the brick and
   *        the floor.
   * @param translational_damping The translational damping coefficient.
   * @param rotational_damping The rotational damping coefficient.
   * @param I_B The rotational inertia of the brick.
   * @param mass_B The mass of the brick.
   */
  InstantaneousContactForceQPController(
      BrickType brick_type, const multibody::MultibodyPlant<double>* plant,
      const Eigen::Ref<const Eigen::Matrix2d>& Kp_tr,
      const Eigen::Ref<const Eigen::Matrix2d>& Kd_tr, double Kp_ro,
      double Kd_ro, double weight_a_error, double weight_thetaddot_error,
      double weight_f_Cb_B, double mu, double translational_damping,
      double rotational_damping, double I_B, double mass_B);

  const systems::InputPort<double>& get_input_port_estimated_plant_state()
      const {
    return this->get_input_port(input_index_estimated_plant_state_);
  }

  // TODO(rcory) The FeedbackController inheritance declares these virtual
  //  functions, but the naming here is too generic, so dispatch to a more
  //  logical name. Perhaps remove the inheritance?
  const systems::InputPort<double>& get_input_port_estimated_state()
  const final {
    return this->get_input_port_estimated_plant_state();
  }

  /**
   * This port only takes the desired brick state, namely
   * [p_y, p_z, θ, ṗ_y, ṗ_z, θ̇]. Please do not connect the finger state to this
   * port.
   */
  const systems::InputPort<double>& get_input_port_desired_brick_state()
      const {
    return this->get_input_port(input_index_desired_brick_state_);
  }
  // TODO(rcory) The FeedbackController inheritance declares these virtual
  //  functions, but the naming here is too generic, so dispatch to a more
  //  logical name. Perhaps remove the inheritance?
  const systems::InputPort<double>& get_input_port_desired_state()
  const final {
    return this->get_input_port_desired_brick_state();
  }

  /**
   * This port takes the desired acceleration of the brick, namely
   * [p̈_y, p̈_z, θ̈], the y/z translational acceleration, and the angle
   * acceleration.
   */
  const systems::InputPort<double>& get_input_port_desired_brick_acceleration()
      const {
    return this->get_input_port(input_index_desired_brick_acceleration_);
  }

  /**
   * This port passes std::unordered_map<Finger, std::pair<BrickFace,
   * Eigen::Vector2d>>, namely it matches a finger to its designated brick face
   * in contact, and the y/z location of the contact point on the brick,
   * expressed in the brick frame.
   */
  const systems::InputPort<double>& get_input_port_finger_face_assignments()
      const {
    return this->get_input_port(input_index_finger_face_assignments_);
  }

  // TODO(rcory) The FeedbackController inheritance declares these virtual
  //  functions, but the naming here is too generic, so dispatch to a more
  //  logical name. Perhaps remove the inheritance?
  const systems::OutputPort<double>& get_output_port_fingers_control() const {
    return this->get_output_port(output_index_fingers_control_);
  }
  const systems::OutputPort<double>& get_output_port_control() const final {
    return this->get_output_port_fingers_control();
  }

  /**
   * This port is used for faking the computed contact force directly to the
   * brick as external spatial wrenches.
   */
  const systems::OutputPort<double>& get_output_port_brick_control() const {
    return this->get_output_port(output_index_brick_control_);
  }

 private:
  void CalcFingersControl(
      const systems::Context<double>& context,
      std::unordered_map<Finger,
                         multibody::ExternallyAppliedSpatialForce<double>>*
          output) const;

  void CalcBrickControl(
      const systems::Context<double>& context,
      std::vector<multibody::ExternallyAppliedSpatialForce<double>>*
          contact_forces) const;

  const BrickType brick_type_;
  const multibody::MultibodyPlant<double>* plant_;
  double mu_;
  Eigen::Matrix2d Kp_tr_;  // Translational proportional QP gain.
  Eigen::Matrix2d Kd_tr_;  // Translational derivative QP gain.
  double Kp_ro_;  // Rotational proportional QP gain.
  double Kd_ro_;  // Rotational derivative QP gain.
  double weight_a_error_;
  double weight_thetaddot_error_;
  double weight_f_Cb_B_;
  double translational_damping_;
  double rotational_damping_;
  double mass_B_;
  double I_B_;

  int brick_translate_y_position_index_;
  int brick_translate_z_position_index_;
  int brick_revolute_x_position_index_;
  multibody::BodyIndex brick_body_index_;

  int input_index_estimated_plant_state_{-1};
  int input_index_desired_brick_state_{-1};
  int input_index_desired_brick_acceleration_{-1};
  int input_index_finger_face_assignments_{-1};
  int output_index_fingers_control_{-1};
  int output_index_brick_control_{-1};
};
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
