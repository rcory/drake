#pragma once

#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

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
   * @param Kp1 The proportional gain of the brick position.
   * @param Kd1 The derivative gain of the brick velocity.
   * @param Kp2 The proportional gain of the brick orientation.
   * @param Kd2 The derivative gain of the brick angular velocity.
   * @param brick_state The brick state include [p_y, p_z, θ, ṗ_y, ṗ_z, θ̇],
   * namely the brick position/orientation and its velocities.
   * @param finger_face_assignment assigns a finger to a contact point on a
   * given face of the brick. finger_face_assignment[finger] gives
   * (face, p_BCb_B), where p_BCb_B is the y/z location of the contact point
   * Cb on the brick frame.
   * @param weight_a The weight of the acceleration error in the cost.
   * @param weight_thetaddot The weight of the angular acceleration error in the
   * cost.
   * @param weight_f_Cb_B The weight of the contact force in the cost.
   */
  InstantaneousContactForceQP(
      const GripperBrickHelper<double>* gripper_brick,
      const Eigen::Ref<const Eigen::Vector2d>& p_WB_planned,
      const Eigen::Ref<const Eigen::Vector2d>& v_WB_planned,
      const Eigen::Ref<const Eigen::Vector2d>& a_WB_planned,
      double theta_planned, double thetadot_planned, double thetaddot_planned,
      const Eigen::Ref<const Eigen::Matrix2d>& Kp1,
      const Eigen::Ref<const Eigen::Matrix2d>& Kd1, double Kp2, double Kd2,
      const Eigen::Ref<const Vector6<double>>& brick_state,
      const std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>&
          finger_face_assignment,
      double weight_a, double weight_thetaddot, double weight_f_Cb);

  const solvers::MathematicalProgram& prog() const { return *prog_; }

  /**
   * Returns (f_Cb_B, p_Cb_B) f_Cb_B is the contact force applied by the finger
   * on the brick, expressed in the brick frame. p_Cb_B is the contact point Cb
   * expressed in the brick frame.
   */
  std::unordered_map<Finger, std::pair<Eigen::Vector2d, Eigen::Vector2d>>
  GetFingerContactForceResult(
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
  const GripperBrickHelper<double>* gripper_brick_;
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
   * @param Kp1 The proportional gain for the brick y/z position.
   * @param Kd1 The derivative gain for the brick y/z position.
   * @param Kp2 The proportional gain for the brick orientation.
   * @param Kd2 The derivative gain for the brick orientation.
   * @param weight_a The weighting for the brick y/z acceleration in the cost.
   * @param weight_thetaddot The weighting for the brick thetaddot in the cost.
   * @param weight_f_Cb_B The weighting for the contact forces in the cost.
   */
  InstantaneousContactForceQPController(
      const GripperBrickHelper<double>* gripper_brick,
      const Eigen::Ref<const Eigen::Matrix2d>& Kp1,
      const Eigen::Ref<const Eigen::Matrix2d>& Kd1, double Kp2, double Kd2,
      double weight_a, double weight_thetaddot, double weight_f_Cb_B);

  const systems::InputPort<double>& get_input_port_estimated_state()
      const final {
    return this->get_input_port(input_index_state_);
  }

  /**
   * This port only takes the desired brick state, namely
   * [p_y, p_z, θ, ṗ_y, ṗ_z, θ̇]. Please do not connect the finger state to this
   * port.
   */
  const systems::InputPort<double>& get_input_port_desired_state() const final {
    return this->get_input_port(input_index_desired_brick_state_);
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
  const systems::InputPort<double>& get_input_port_finger_contact() const {
    return this->get_input_port(input_index_finger_contact_);
  }

  const systems::OutputPort<double>& get_output_port_control() const final {
    return this->get_output_port(output_index_control_);
  }

  /**
   * This port is used for faking the computed contact force directly to the
   * brick as external spatial wrenches.
   */
  const systems::OutputPort<double>& get_output_port_contact_force() const {
    return this->get_output_port(output_index_contact_force_);
  }

 private:
  void CalcControl(
      const systems::Context<double>& context,
      std::unordered_map<Finger,
                         multibody::ExternallyAppliedSpatialForce<double>>*
          output) const;

  void CalcSpatialContactForce(
      const systems::Context<double>& context,
      std::vector<multibody::ExternallyAppliedSpatialForce<double>>*
          contact_forces) const;

  const GripperBrickHelper<double>* gripper_brick_;
  Eigen::Matrix2d Kp1_;
  Eigen::Matrix2d Kd1_;
  double Kp2_;
  double Kd2_;
  double weight_a_;
  double weight_thetaddot_;
  double weight_f_Cb_B_;

  int input_index_state_{-1};
  int input_index_desired_brick_state_{-1};
  int input_index_desired_brick_acceleration_{-1};
  int input_index_p_BFingerTip_{-1};
  int input_index_finger_contact_{-1};
  int output_index_control_{-1};
  int output_index_contact_force_{-1};
};
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
