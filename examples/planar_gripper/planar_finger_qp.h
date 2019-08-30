#pragma once

#include <memory>

#include "drake/examples/planar_gripper/gripper_brick.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"

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
   * @param plant_context The context containing the system current q and v.
   * @param weight_thetaddot_error The weight of the thetaddot error in the
   * cost.
   * @param weight_f_Cb The weight of the contact force in the cost.
   * @param contact_face The brick face that is in contact with the finger.
   * @param mu the friction coefficient between the finger tip and the brick.
   * @param p_L2FingerTip The position of the finger tip (the center of the
   * bubble on the tip of the finger), expressed in the finger link 2 frame.
   * @param I_B The inertia of the brick.
   * @param finger_tip_radius The radius of the bubble on the finger.
   */
  PlanarFingerInstantaneousQP(const multibody::MultibodyPlant<double>* plant,
                              double theta_planned, double thetadot_planned,
                              double thetaddot_planned, double Kp, double Kd,
                              const systems::Context<double>& plant_context,
                              double weight_thetaddot_error, double weight_f_Cb,
                              BrickFace contact_face, double mu,
                              const Eigen::Vector3d& p_L2FingerTip, double I_B,
                              double finger_tip_radius);

  const solvers::MathematicalProgram& prog() const { return *prog_; }

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
};
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
