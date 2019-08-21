#pragma once

#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/examples/planar_gripper/gripper_brick.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"

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
   * @param plant_context The system's current position/velocity.
   * @param weight_a The weight of the acceleration error in the cost.
   * @param weight_thetaddot The weight of the angular acceleration error in the
   * cost.
   * @param weight_f_Cb_B The weight of the contact force in the cost.
   * @param finger_face_assignment The assignment between the finger/face in
   * contact.
   */
  InstantaneousContactForceQP(
      const GripperBrickHelper<double>* gripper_brick,
      const Eigen::Ref<const Eigen::Vector2d>& p_WB_planned,
      const Eigen::Ref<const Eigen::Vector2d>& v_WB_planned,
      const Eigen::Ref<const Eigen::Vector2d>& a_WB_planned,
      double theta_planned, double thetadot_planned, double thetaddot_planned,
      const Eigen::Ref<const Eigen::Matrix2d>& Kp1,
      const Eigen::Ref<const Eigen::Matrix2d>& Kd1, double Kp2, double Kd2,
      const systems::Context<double>& plant_context, double weight_a,
      double weight_thetaddot, double weight_f_Cb,
      const std::map<Finger, BrickFace>& finger_face_assignment);

  const solvers::MathematicalProgram& prog() const { return *prog_; }

  /**
   * Returns f_Cb_B (The contact force applied by the finger on the brick,
   * expressed in the brick frame).
   */
  std::unordered_map<Finger, Eigen::Vector2d> GetFingerContactForceResult(
      const solvers::MathematicalProgramResult& result) const;

 private:
  struct FingerFaceContact {
    FingerFaceContact(Finger m_finger, BrickFace m_brick_face,
                      const Vector2<symbolic::Variable>& m_f_Cb_B_edges,
                      double m_mu)
        : finger{m_finger},
          brick_face{m_brick_face},
          f_Cb_B_edges{m_f_Cb_B_edges},
          mu{m_mu} {}
    Finger finger;
    BrickFace brick_face;
    Vector2<symbolic::Variable> f_Cb_B_edges;
    double mu;
  };
  const GripperBrickHelper<double>* gripper_brick_;
  std::unique_ptr<solvers::MathematicalProgram> prog_;
  std::vector<FingerFaceContact> finger_face_contacts_;
};
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
