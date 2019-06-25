#pragma once

#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/binding.h"
#include "drake/solvers/constraint.h"

namespace drake {
namespace multibody {

/**
 * Impose the semi-implicit Euler constraint:
 * 0 = (qₙ₊₁ - qₙ) - Vₙ₊₁ * dt
 */
class SemiImplicitEulerConstraint final : public solvers::Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SemiImplicitEulerConstraint)

  SemiImplicitEulerConstraint(const MultibodyPlant<AutoDiffXd>* plant,
                              systems::Context<AutoDiffXd>* context,
                              double fixed_timestep);

  ~SemiImplicitEulerConstraint() override {}

 private:

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const final;

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const final;

  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
              VectorX<symbolic::Expression>* y) const final;

  const MultibodyPlant<AutoDiffXd>* const plant_;
  systems::Context<AutoDiffXd>* const context_;
  AutoDiffXd evaluation_time_{0};
  const double fixed_timestep_;
};

}  // namespace drake
}  // namespace multibody