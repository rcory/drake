#pragma once

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "drake/geometry/scene_graph.h"
#include "drake/multibody/optimization/contact_wrench_evaluator.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/binding.h"
#include "drake/solvers/constraint.h"
#include "drake/multibody/optimization/static_equilibrium_constraint.h"

namespace drake {
namespace multibody {

/**
 * Impose the contact implicit constraint:
 * 0 = (qₙ₊₁ - qₙ) - Vₙ₊₁ * dt
 * 0 = (Buₙ₊₁ + ∑J_WBᵀ(qₙ₊₁) * Fₙ₊₁_B_W + tau_g(qₙ₊₁) - C(qₙ₊₁, Vₙ₊₁)) * dt
 *     - M(qₙ₊₁) * (Vₙ₊₁ - Vₙ)
 */
class ContactImplicitConstraint final : public solvers::Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ContactImplicitConstraint)

  /**
   * Create the contact implicit constraints
   * 0 = (B*u[n+1]  + ∑ᵢ Jᵢ(q[n+1])ᵀFᵢ_AB_W(λᵢ[n+1]) + g(q[n+1])
   *     - C(q[n+1], v[n+1])) * dt
   *     - M(q[n+1])(v[n+1] - v[n])
   * 0 = (q[n+1] - q[n]) - v[n+1] * dt
   * These constraints depend on the variables
   * q[n], v[n], q[n+1], v[n+1], u[n+1] and λ[n+1] ==
   * {state[n], state[n+1], u[n+1], lambda[n+1]}.
   * @param plant The plant on which the constraint is imposed.
   * @param context The context for the subsystem @p plant.
   * @param contact_wrench_evaluators_and_lambda For each contact pair, we
   * need to compute the contact wrench applied at the point of contact,
   * expressed in the world frame, namely Fᵢ_AB_W(λᵢ).
   * `contact_wrench_evaluators_and_lambda.first` is the evaluator for computing
   * this contact wrench from the variables λᵢ.
   * `contact_wrench_evaluators_and_lambda.second` are the decision variable λᵢ
   * used in computing the contact wrench. Notice the generalized position `q`
   * is not included in variables contact_wrench_evaluators_and_lambda.second.
   * @param state_vars The decision variables for [q[n]; v[n]].
   * @param state_next_vars The decision variables for [q[n+1]; v[n+1]].
   * @param u_next_vars The decision variables for u[n+1].
   * @return binding The binding between the contact implicit constraint and
   * the variables q[n], v[n], q[n+1], v[n+1], u[n+1] and λ[n+1].
   * @pre @p plant must have been connected to a SceneGraph properly. You could
   * refer to AddMultibodyPlantSceneGraph on how to connect a MultibodyPlant to
   * a SceneGraph.
   */
  static solvers::Binding<ContactImplicitConstraint> MakeBinding(
      const MultibodyPlant<AutoDiffXd>* plant,
      systems::Context<AutoDiffXd>* context,
      const std::vector<std::pair<std::shared_ptr<ContactWrenchEvaluator>,
                                  VectorX<symbolic::Variable>>>&
          contact_wrench_evaluators_and_lambda,
      const Eigen::Ref<const VectorX<symbolic::Variable>>& state_vars,
      const Eigen::Ref<const VectorX<symbolic::Variable>>& state_next_vars,
      const Eigen::Ref<const VectorX<symbolic::Variable>>& u_next_vars);

  ~ContactImplicitConstraint() override {}

  /**
   * Getter for contact_pair_to_wrench_evaluator, passed in the constructor.
   */
  const std::map<SortedPair<geometry::GeometryId>,
                 internal::GeometryPairContactWrenchEvaluatorBinding>&
  contact_pair_to_wrench_evaluator() const {
    return contact_pair_to_wrench_evaluator_;
  }

 private:
  /**
   * The user cannot call this constructor, as it is inconvenient to do so.
   * The user must call MakeBinding() to construct a
   * StaticEquilibriumConstraint.
   */
  ContactImplicitConstraint(
      const MultibodyPlant<AutoDiffXd>* plant,
      systems::Context<AutoDiffXd>* context,
      const std::map<SortedPair<geometry::GeometryId>,
                     internal::GeometryPairContactWrenchEvaluatorBinding>&
          contact_pair_to_wrench_evaluator);

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const final;
  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const final;
  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
              VectorX<symbolic::Expression>* y) const final;

  const MultibodyPlant<AutoDiffXd>* const plant_;
  systems::Context<AutoDiffXd>* const context_;
  const std::map<SortedPair<geometry::GeometryId>,
                 internal::GeometryPairContactWrenchEvaluatorBinding>
      contact_pair_to_wrench_evaluator_;
  const MatrixX<AutoDiffXd> B_actuation_;
};

}  // namespace multibody
}  // namespace drake
