#include "drake/multibody/optimization/semi_implicit_euler_constraint.h"

#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/binding.h"
#include "drake/solvers/constraint.h"

namespace drake {
namespace multibody {

/**
 * Impose the semi-implicit Euler constraint:
 * 0 = (qₙ₊₁ - qₙ) - Vₙ₊₁ * dt
 */
 // TODO(rcory) The above equation assumes size(q) == size(v).
SemiImplicitEulerConstraint::SemiImplicitEulerConstraint(
    const MultibodyPlant<AutoDiffXd>* plant,
    systems::Context<AutoDiffXd>* context, double fixed_timestep)
    : solvers::Constraint(plant->num_velocities(), 3 * plant->num_velocities(),
                          Eigen::VectorXd::Zero(plant->num_velocities()),
                          Eigen::VectorXd::Zero(plant->num_velocities())),
  plant_{plant},
  context_{context},
  fixed_timestep_(fixed_timestep) {
  if (fixed_timestep_ <= 0) {
    throw std::logic_error(
        "SemiImplicitEulerConstraint: time step must be greater than zero.");
  }
}

void SemiImplicitEulerConstraint::DoEval(
    const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd* y) const {
  AutoDiffVecXd y_autodiff(num_constraints());
  DoEval(x.cast<AutoDiffXd>(), &y_autodiff);
  *y = math::autoDiffToValueMatrix(y_autodiff);
}

// The format of the input to the eval() function is a vector
// containing {q[n], q[n+1], v[n+1]}
void SemiImplicitEulerConstraint::DoEval(
    const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd* y) const {
  // TODO(rcory) this assert assumes size(q) == size(v)
  DRAKE_ASSERT(plant_->num_positions() == plant_->num_velocities());
  DRAKE_ASSERT(x.size() == 3 * plant_->num_velocities());

  // Extract the input variables
  const auto q = x.head(plant_->num_positions());
  const auto q_next = x.segment(plant_->num_positions(), plant_->num_positions());
  const auto v_next = x.tail(plant_->num_positions());

  *y = (q_next - q) - v_next * fixed_timestep_;
}

void SemiImplicitEulerConstraint::DoEval(
    const Eigen::Ref<const VectorX<symbolic::Variable>>&,
    VectorX<symbolic::Expression>*) const {
  throw std::logic_error(
      "SemiImplicitEulerConstraint does not support symbolic evaluation.");
}

}  // namespace drake
}  // namespace multibody