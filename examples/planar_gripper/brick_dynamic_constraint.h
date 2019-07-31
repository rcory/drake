#pragma once

#include <map>

#include "drake/examples/planar_gripper/gripper_brick.h"
#include "drake/solvers/constraint.h"

namespace drake {
namespace examples {
namespace planar_gripper {
/**
 * Compute the total wrench on the brick.
 *
 *     mg + f_friction + ∑ R_WB * f_FiB_B
 *     τ_friction + ∑ p_BCbi * f_FiB_B
 * where f_friction is the friction force between the brick and the lid, and
 * τ_friction is the friction torque between the brick and the lid.
 * This evaluator is going to be used in enforcing the dynamic constraint on the
 * brick.
 */
class BrickTotalWrenchEvaluator : public solvers::EvaluatorBase {
 public:
  BrickTotalWrenchEvaluator(const GripperBrickHelper<double>* gripper_brick,
                            systems::Context<double>* plant_context,
                            std::map<Finger, BrickFace> finger_faces,
                            double brick_lid_friction_force_magnitude,
                            double brick_lid_friction_torque_magnitude);

  const std::map<Finger, BrickFace>& finger_faces() const {
    return finger_faces_;
  }

  template <typename T>
  void ComposeX(const Eigen::Ref<const VectorX<T>>& q,
                const T& v_brick_translation_y, const T& v_brick_translation_z,
                const T& v_brick_rotation_x, const Matrix2X<T>& f_FB_B,
                VectorX<T>* x) const {
    x->resize(num_vars());
    x->head(gripper_brick_->plant().num_positions()) = q;
    (*x)(q.rows()) = v_brick_translation_y;
    (*x)(q.rows() + 1) = v_brick_translation_z;
    (*x)(q.rows() + 2) = v_brick_rotation_x;
    for (int i = 0; i < f_FB_B.cols(); ++i) {
      x->template segment<2>(q.rows() + 3 + 2 * i) = f_FB_B.col(i);
    }
  }

 private:
  template <typename T>
  void DecomposeX(const Eigen::Ref<const VectorX<T>>& x, VectorX<T>* q,
                  T* v_brick_translation_y, T* v_brick_translation_z,
                  T* v_brick_rotation_x, Matrix2X<T>* f_FB_B) const {
    *q = x.head(gripper_brick_->plant().num_positions());
    *v_brick_translation_y = x(gripper_brick_->plant().num_positions());
    *v_brick_translation_z = x(gripper_brick_->plant().num_positions() + 1);
    *v_brick_rotation_x = x(gripper_brick_->plant().num_positions() + 2);
    f_FB_B->resize(2, static_cast<int>(finger_faces_.size()));
    for (int i = 0; i < f_FB_B->cols(); ++i) {
      f_FB_B->col(i) = x.template segment<2>(
          gripper_brick_->plant().num_positions() + 3 + 2 * i);
    }
  }

  template <typename T>
  void DoEvalGeneric(const Eigen::Ref<const VectorX<T>>& x,
                     VectorX<T>* y) const;

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override;
  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override;
  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>&,
              VectorX<symbolic::Expression>*) const override {
    throw std::runtime_error(
        "BrickTotalWrenchEvaluator::DoEval doesn't support symbolic "
        "variables.");
  }

  const GripperBrickHelper<double>* const gripper_brick_;
  systems::Context<double>* plant_context_;
  std::map<Finger, BrickFace> finger_faces_;
  double brick_lid_friction_force_magnitude_;
  double brick_lid_friction_torque_magnitude_;
};

/**
 * Enforce the backward Euler integration constraint
 * <pre>
 * m(v̇[n+1] - v̇[n]) = (mg + f_friction[n+1] + ∑ R_WB[n+1] * f_FiB_B[n+1]) * dt
 * I(θ_dot[n+1] - θ_dot[n]) = (τ_friction[n+1] + ∑ p_BCbi[n+1] * f_FiB_B[n+1]) *
 * dt
 * </pre>
 * where f_friction is the friction force between the * brick
 * and the lid, and τ_friction is the friction torque between the brick and the
 * lid. The decision variables are q[n+1], v_brick[n+1], v_brick[n], f_FB_B[n+1]
 * and dt.
 */
class BrickDynamicBackwardEulerConstraint : public solvers::Constraint {
 public:
  BrickDynamicBackwardEulerConstraint(
      const GripperBrickHelper<double>* gripper_brick,
      systems::Context<double>* plant_context,
      std::map<Finger, BrickFace> finger_faces,
      double brick_lid_friction_force_magnitude,
      double brick_lid_friction_torque_magnitude);

  ~BrickDynamicBackwardEulerConstraint() override {}

  template <typename T>
  void ComposeX(const Eigen::Ref<const VectorX<T>>& q_r,
                const T& v_brick_r_translation_y,
                const T& v_brick_r_translation_z, const T& v_brick_r_rotation_x,
                const T& v_brick_l_translation_y,
                const T& v_brick_l_translation_z, const T& v_brick_l_rotation_x,
                const Matrix2X<T>& f_FB_B, const T& dt, VectorX<T>* x) const {
    x->resize(num_vars());
    x->head(gripper_brick_->plant().num_positions()) = q_r;
    (*x)(q_r.rows()) = v_brick_r_translation_y;
    (*x)(q_r.rows() + 1) = v_brick_r_translation_z;
    (*x)(q_r.rows() + 2) = v_brick_r_rotation_x;
    (*x)(q_r.rows() + 3) = v_brick_l_translation_y;
    (*x)(q_r.rows() + 4) = v_brick_l_translation_z;
    (*x)(q_r.rows() + 5) = v_brick_l_rotation_x;
    for (int i = 0; i < f_FB_B.cols(); ++i) {
      x->template segment<2>(q_r.rows() + 6 + 2 * i) = f_FB_B.col(i);
    }
    (*x)(num_vars() - 1) = dt;
  }

 private:
  template <typename T>
  void DecomposeX(const Eigen::Ref<const VectorX<T>>& x, VectorX<T>* q_r,
                  T* v_brick_r_translation_y, T* v_brick_r_translation_z,
                  T* v_brick_r_rotation_x, T* v_brick_l_translation_y,
                  T* v_brick_l_translation_z, T* v_brick_l_rotation_x,
                  Matrix2X<T>* f_FB_B, T* dt) const {
    *q_r = x.head(gripper_brick_->plant().num_positions());
    *v_brick_r_translation_y = x(q_r->rows());
    *v_brick_r_translation_z = x(q_r->rows() + 1);
    *v_brick_r_rotation_x = x(q_r->rows() + 2);
    *v_brick_l_translation_y = x(q_r->rows() + 3);
    *v_brick_l_translation_z = x(q_r->rows() + 4);
    *v_brick_l_rotation_x = x(q_r->rows() + 5);
    f_FB_B->resize(2, wrench_evaluator_.finger_faces().size());
    for (int i = 0; i < f_FB_B->cols(); ++i) {
      f_FB_B->col(i) = x.template segment<2>(q_r->rows() + 6 + 2 * i);
    }
    *dt = x(x.rows() - 1);
  }

  template <typename T>
  void DoEvalGeneric(const Eigen::Ref<const VectorX<T>>& x,
                     VectorX<T>* y) const;

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override;
  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override;
  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>&,
              VectorX<symbolic::Expression>*) const override {
    throw std::runtime_error(
        "BrickDynamicConstraint::DoEval doesn't support symbolic variables.");
  }

  const GripperBrickHelper<double>* const gripper_brick_;
  systems::Context<double>* plant_context_;
  BrickTotalWrenchEvaluator wrench_evaluator_;
};

/**
 * Enforce the midpoint integration constraint
 *
 *     m(v̇[n+1] - v̇[n]) = (f[n] + f[n+1]) / 2 * dt
 *     I(θ_dot[n+1] - θ_dot[n]) = (τ[n] + τ[n+1]) / 2 * dt
 * where
 *
 *     f[n] = mg + f_friction[n] + ∑ R_WB[n] * f_FiB_B[n]
 *     τ[n] = τ_friction[n] + ∑ p_BCbi[n] * f_FiB_B[n]
 * where f_friction is the friction force between the brick and the lid, and
 * τ_friction is the friction torque between the brick and the lid.
 * The decision variables are (q[n+1], q[n], v_brick[n+1], v_brick[n],
 * f_FB_B[n+1], f_FB_B[n], dt).
 */
class BrickDynamicMidpointIntegrationConstraint : public solvers::Constraint {
 public:
  /**
   * @param gripper_brick @note gripper_brick should be alive during the life
   * span of this constraint.
   * @param plant_context_r The context to store q[n+1], v[n+1]
   * @param plant_context_l The context to store q[n], v[n]
   */
  BrickDynamicMidpointIntegrationConstraint(
      const GripperBrickHelper<double>* gripper_brick,
      systems::Context<double>* plant_context_r,
      systems::Context<double>* plant_context_l,
      std::map<Finger, BrickFace> finger_faces_r,
      std::map<Finger, BrickFace> finger_faces_l,
      double brick_lid_friction_force_magnitude,
      double brick_lid_friction_torque_magnitude);

  ~BrickDynamicMidpointIntegrationConstraint() override {}

  template <typename T>
  void ComposeX(const Eigen::Ref<const VectorX<T>>& q_r,
                const Eigen::Ref<const VectorX<T>>& q_l,
                const T& v_brick_r_translation_y,
                const T& v_brick_r_translation_z, const T& v_brick_r_rotation_x,
                const T& v_brick_l_translation_y,
                const T& v_brick_l_translation_z, const T& v_brick_l_rotation_x,
                const Matrix2X<T>& f_FB_B_r, const Matrix2X<T>& f_FB_B_l,
                const T& dt, VectorX<T>* x) const {
    x->resize(num_vars());
    const int nq = gripper_brick_->plant().num_positions();
    x->head(nq) = q_r;
    x->segment(nq, nq) = q_l;
    (*x)(2 * nq) = v_brick_r_translation_y;
    (*x)(2 * nq + 1) = v_brick_r_translation_z;
    (*x)(2 * nq + 2) = v_brick_r_rotation_x;
    (*x)(2 * nq + 3) = v_brick_l_translation_y;
    (*x)(2 * nq + 4) = v_brick_l_translation_z;
    (*x)(2 * nq + 5) = v_brick_l_rotation_x;
    for (int i = 0; i < f_FB_B_r.cols(); ++i) {
      x->template segment<2>(2 * nq + 6 + 2 * i) = f_FB_B_r.col(i);
    }
    for (int i = 0; i < f_FB_B_l.cols(); ++i) {
      x->template segment<2>(2 * nq + 6 + 2 * f_FB_B_r.cols() + 2 * i) =
          f_FB_B_l.col(i);
    }
    (*x)(num_vars() - 1) = dt;
  }

 private:
  template <typename T>
  void DecomposeX(const Eigen::Ref<const VectorX<T>>& x, VectorX<T>* q_r,
                  VectorX<T>* q_l, T* v_brick_r_translation_y,
                  T* v_brick_r_translation_z, T* v_brick_r_rotation_x,
                  T* v_brick_l_translation_y, T* v_brick_l_translation_z,
                  T* v_brick_l_rotation_x, Matrix2X<T>* f_FB_B_r,
                  Matrix2X<T>* f_FB_B_l, T* dt) const {
    const int nq = gripper_brick_->plant().num_positions();
    *q_r = x.head(nq);
    *q_l = x.segment(nq, nq);
    *v_brick_r_translation_y = x(2 * nq);
    *v_brick_r_translation_z = x(2 * nq + 1);
    *v_brick_r_rotation_x = x(2 * nq + 2);
    *v_brick_l_translation_y = x(2 * nq + 3);
    *v_brick_l_translation_z = x(2 * nq + 4);
    *v_brick_l_rotation_x = x(2 * nq + 5);
    f_FB_B_r->resize(2, wrench_evaluator_r_.finger_faces().size());
    f_FB_B_l->resize(2, wrench_evaluator_l_.finger_faces().size());
    for (int i = 0; i < f_FB_B_r->cols(); ++i) {
      f_FB_B_r->col(i) = x.template segment<2>(2 * nq + 6 + 2 * i);
    }
    for (int i = 0; i < f_FB_B_l->cols(); ++i) {
      f_FB_B_l->col(i) =
          x.template segment<2>(2 * nq + 6 + 2 * f_FB_B_r->cols() + 2 * i);
    }
    *dt = x(x.rows() - 1);
  }

  template <typename T>
  void DoEvalGeneric(const Eigen::Ref<const VectorX<T>>& x,
                     VectorX<T>* y) const;

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override;
  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override;
  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>&,
              VectorX<symbolic::Expression>*) const override {
    throw std::runtime_error(
        "BrickDynamicConstraint::DoEval doesn't support symbolic variables.");
  }

  const GripperBrickHelper<double>* const gripper_brick_;
  systems::Context<double>* plant_context_r_;
  systems::Context<double>* plant_context_l_;
  BrickTotalWrenchEvaluator wrench_evaluator_r_;
  BrickTotalWrenchEvaluator wrench_evaluator_l_;
};
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
