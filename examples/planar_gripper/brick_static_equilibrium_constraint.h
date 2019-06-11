#pragma once

#include "drake/examples/planar_gripper/gripper_brick.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/constraint.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace examples {
/** Given the set of contacts between the fingers and the brick, impose the
 * static equilibrium as a nonlinear constraint, that the total force/torque
 * applied on the brick is 0.
 */
class BrickStaticEquilibriumNonlinearConstraint : public solvers::Constraint {
 public:
  BrickStaticEquilibriumNonlinearConstraint(
      const GripperBrickSystem<double>& gripper_brick_system,
      std::vector<std::pair<Finger, BrickFace>> finger_face_contacts,
      systems::Context<double>* plant_mutable_context);

 private:
  Eigen::Vector3d ComputeFingerTipInBrickFrame(
      const multibody::MultibodyPlant<double>& plant, const Finger finger,
      const systems::Context<double>& plant_context,
      const Eigen::Ref<const Eigen::VectorXd>&) const;

  Vector3<AutoDiffXd> ComputeFingerTipInBrickFrame(
      const multibody::MultibodyPlant<double>& plant, const Finger finger,
      const systems::Context<double>& plant_context,
      const Eigen::Ref<const AutoDiffVecXd>& q) const;

  template <typename T>
  void DoEvalGeneric(const Eigen::Ref<const VectorX<T>>& x,
                     VectorX<T>* y) const;

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const;

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd* y) const;

  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>&,
              VectorX<symbolic::Expression>*) const;

  const GripperBrickSystem<double>& gripper_brick_system_;
  double brick_mass_;
  std::vector<std::pair<Finger, BrickFace>> finger_face_contacts_;
  systems::Context<double>* plant_mutable_context_;
};

void AddBrickStaticEquilibriumConstraint(
    const GripperBrickSystem<double>& gripper_brick_system,
    const std::vector<std::pair<Finger, BrickFace>>& finger_face_contacts,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& q_vars,
    systems::Context<double>* plant_mutable_context,
    solvers::MathematicalProgram* prog);
}  // namespace examples
}  // namespace drake
