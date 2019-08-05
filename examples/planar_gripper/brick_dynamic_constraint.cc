#include "drake/examples/planar_gripper/brick_dynamic_constraint.h"

#include <utility>

#include "drake/examples/planar_gripper/gripper_brick_planning_constraint_helper.h"
#include "drake/multibody/inverse_kinematics/kinematic_constraint_utilities.h"

namespace drake {
namespace examples {
namespace planar_gripper {
BrickTotalWrenchEvaluator::BrickTotalWrenchEvaluator(
    const GripperBrickHelper<double>* gripper_brick,
    systems::Context<double>* plant_context,
    std::map<Finger, BrickFace> finger_faces,
    double brick_lid_friction_force_magnitude,
    double brick_lid_friction_torque_magnitude)
    : solvers::EvaluatorBase(3, gripper_brick->plant().num_positions() + 3 +
                                    2 * static_cast<int>(finger_faces.size())),
      gripper_brick_{gripper_brick},
      plant_context_{plant_context},
      finger_faces_{std::move(finger_faces)},
      brick_lid_friction_force_magnitude_{brick_lid_friction_force_magnitude},
      brick_lid_friction_torque_magnitude_{
          brick_lid_friction_torque_magnitude} {}

template <typename T>
void BrickTotalWrenchEvaluator::DoEvalGeneric(
    const Eigen::Ref<const VectorX<T>>& x, VectorX<T>* y) const {
  VectorX<T> q;
  T v_brick_translation_y, v_brick_translation_z, v_brick_rotation_x;
  Matrix2X<T> f_FB_B;
  DecomposeX<T>(x, &q, &v_brick_translation_y, &v_brick_translation_z,
                &v_brick_rotation_x, &f_FB_B);
  y->resize(3);
  y->template head<2>() =
      gripper_brick_->brick_frame().body().get_default_mass() *
      Eigen::Vector2d(0, -9.81);
  // Compute a "unit length" vector along the brick translational velocity.
  using std::pow;
  using std::sqrt;
  Vector2<T> vhat_brick_translation =
      Vector2<T>(v_brick_translation_y, v_brick_translation_z) /
      sqrt(pow(v_brick_translation_y, 2) + pow(v_brick_translation_z, 2) +
           1E-6 /* Add 1E-6 to make sqrt function differentiable */);
  y->template head<2>() -=
      brick_lid_friction_force_magnitude_ * vhat_brick_translation;
  using std::cos;
  using std::sin;
  const T theta = q(gripper_brick_->brick_revolute_x_position_index());
  const T cos_theta = cos(theta);
  const T sin_theta = sin(theta);
  Matrix2<T> R_WB;
  R_WB << cos_theta, -sin_theta, sin_theta, cos_theta;
  y->template head<2>() += R_WB * f_FB_B.rowwise().sum();
  // We use v / sqrt(v*v + epsilon) as a smoothed version of sign(v) function.
  (*y)(2) = -v_brick_rotation_x / sqrt(pow(v_brick_rotation_x, 2) + 1E-6) *
            brick_lid_friction_torque_magnitude_;
  multibody::internal::UpdateContextConfiguration(plant_context_,
                                                  gripper_brick_->plant(), q);
  int contact_count = 0;
  for (const auto& finger_face : finger_faces_) {
    const Vector3<T> p_BTip = ComputeFingerTipInBrickFrame(
        *gripper_brick_, finger_face.first, *plant_context_, q);
    // C is the point of contact between the finger and the brick.
    Vector2<T> p_BC = p_BTip.template tail<2>();
    switch (finger_face.second) {
      case BrickFace::kPosY: {
        p_BC(0) -= T(gripper_brick_->finger_tip_radius());
        break;
      }
      case BrickFace::kNegY: {
        p_BC(0) += T(gripper_brick_->finger_tip_radius());
        break;
      }
      case BrickFace::kPosZ: {
        p_BC(1) -= T(gripper_brick_->finger_tip_radius());
        break;
      }
      case BrickFace::kNegZ: {
        p_BC(1) += T(gripper_brick_->finger_tip_radius());
        break;
      }
    }
    (*y)(2) +=
        p_BC(0) * f_FB_B(1, contact_count) - p_BC(1) * f_FB_B(0, contact_count);
    contact_count++;
  }
}

void BrickTotalWrenchEvaluator::DoEval(
    const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd* y) const {
  DoEvalGeneric<double>(x, y);
}

void BrickTotalWrenchEvaluator::DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
                                       AutoDiffVecXd* y) const {
  DoEvalGeneric<AutoDiffXd>(x, y);
}

BrickDynamicBackwardEulerConstraint::BrickDynamicBackwardEulerConstraint(
    const GripperBrickHelper<double>* const gripper_brick,
    systems::Context<double>* plant_context,
    std::map<Finger, BrickFace> finger_faces,
    double brick_lid_friction_force_magnitude,
    double brick_lid_friction_torque_magnitude)
    : solvers::Constraint(3,
                          gripper_brick->plant().num_positions() + 6 +
                              2 * static_cast<int>(finger_faces.size()) + 1,
                          Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()),
      gripper_brick_(gripper_brick),
      plant_context_{plant_context},
      wrench_evaluator_(gripper_brick, plant_context, finger_faces,
                        brick_lid_friction_force_magnitude,
                        brick_lid_friction_torque_magnitude) {}

template <typename T>
void BrickDynamicBackwardEulerConstraint::DoEvalGeneric(
    const Eigen::Ref<const VectorX<T>>& x, VectorX<T>* y) const {
  VectorX<T> q_r;
  T v_brick_r_translation_y, v_brick_r_translation_z, v_brick_r_rotation_x,
      v_brick_l_translation_y, v_brick_l_translation_z, v_brick_l_rotation_x;
  Matrix2X<T> f_FB_B;
  T dt;
  DecomposeX(x, &q_r, &v_brick_r_translation_y, &v_brick_r_translation_z,
             &v_brick_r_rotation_x, &v_brick_l_translation_y,
             &v_brick_l_translation_z, &v_brick_l_rotation_x, &f_FB_B, &dt);
  VectorX<T> total_wrench;
  VectorX<T> wrench_evaluator_x;
  wrench_evaluator_.ComposeX<T>(q_r, v_brick_r_translation_y,
                                v_brick_r_translation_z, v_brick_r_rotation_x,
                                f_FB_B, &wrench_evaluator_x);
  wrench_evaluator_.Eval(wrench_evaluator_x, &total_wrench);
  y->resize(3);
  y->template head<2>() =
      gripper_brick_->brick_frame().body().get_default_mass() *
          Vector2<T>(v_brick_r_translation_y - v_brick_l_translation_y,
                     v_brick_r_translation_z - v_brick_l_translation_z) -
      total_wrench.template head<2>() * dt;
  multibody::internal::UpdateContextConfiguration(plant_context_,
                                                  gripper_brick_->plant(), q_r);
  const double I = gripper_brick_->brick_frame()
                       .body()
                       .CalcSpatialInertiaInBodyFrame(*plant_context_)
                       .CalcRotationalInertia()
                       .get_moments()(0);
  (*y)(2) =
      I * (v_brick_r_rotation_x - v_brick_l_rotation_x) - total_wrench(2) * dt;
}

void BrickDynamicBackwardEulerConstraint::DoEval(
    const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd* y) const {
  DoEvalGeneric<double>(x, y);
}

void BrickDynamicBackwardEulerConstraint::DoEval(
    const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd* y) const {
  DoEvalGeneric<AutoDiffXd>(x, y);
}

BrickDynamicMidpointIntegrationConstraint::
    BrickDynamicMidpointIntegrationConstraint(
        const GripperBrickHelper<double>* const gripper_brick,
        systems::Context<double>* plant_context_r,
        systems::Context<double>* plant_context_l,
        std::map<Finger, BrickFace> finger_faces_r,
        std::map<Finger, BrickFace> finger_faces_l,
        double brick_lid_friction_force_magnitude,
        double brick_lid_friction_torque_magnitude)
    : solvers::Constraint(3,
                          gripper_brick->plant().num_positions() * 2 + 6 +
                              2 * static_cast<int>(finger_faces_l.size()) +
                              2 * static_cast<int>(finger_faces_r.size()) + 1,
                          Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()),
      gripper_brick_(gripper_brick),
      plant_context_r_{plant_context_r},
      plant_context_l_{plant_context_l},
      wrench_evaluator_r_(gripper_brick, plant_context_r_, finger_faces_r,
                          brick_lid_friction_force_magnitude,
                          brick_lid_friction_torque_magnitude),
      wrench_evaluator_l_(gripper_brick, plant_context_l_, finger_faces_l,
                          brick_lid_friction_force_magnitude,
                          brick_lid_friction_torque_magnitude) {}

template <typename T>
void BrickDynamicMidpointIntegrationConstraint::DoEvalGeneric(
    const Eigen::Ref<const VectorX<T>>& x, VectorX<T>* y) const {
  VectorX<T> q_r, q_l;
  T v_brick_r_translation_y, v_brick_r_translation_z, v_brick_r_rotation_x,
      v_brick_l_translation_y, v_brick_l_translation_z, v_brick_l_rotation_x;
  Matrix2X<T> f_FB_B_r, f_FB_B_l;
  T dt;
  DecomposeX(x, &q_r, &q_l, &v_brick_r_translation_y, &v_brick_r_translation_z,
             &v_brick_r_rotation_x, &v_brick_l_translation_y,
             &v_brick_l_translation_z, &v_brick_l_rotation_x, &f_FB_B_r,
             &f_FB_B_l, &dt);
  VectorX<T> total_wrench_r, total_wrench_l;
  VectorX<T> wrench_evaluator_r_x, wrench_evaluator_l_x;
  wrench_evaluator_r_.ComposeX<T>(q_r, v_brick_r_translation_y,
                                  v_brick_r_translation_z, v_brick_r_rotation_x,
                                  f_FB_B_r, &wrench_evaluator_r_x);
  wrench_evaluator_l_.ComposeX<T>(q_l, v_brick_l_translation_y,
                                  v_brick_l_translation_z, v_brick_l_rotation_x,
                                  f_FB_B_l, &wrench_evaluator_l_x);
  wrench_evaluator_r_.Eval(wrench_evaluator_r_x, &total_wrench_r);
  wrench_evaluator_l_.Eval(wrench_evaluator_l_x, &total_wrench_l);
  y->resize(3);
  y->template head<2>() =
      gripper_brick_->brick_frame().body().get_default_mass() *
          Vector2<T>(v_brick_r_translation_y - v_brick_l_translation_y,
                     v_brick_r_translation_z - v_brick_l_translation_z) -
      (total_wrench_r.template head<2>() + total_wrench_l.template head<2>()) /
          2 * dt;
  multibody::internal::UpdateContextConfiguration(plant_context_r_,
                                                  gripper_brick_->plant(), q_r);
  multibody::internal::UpdateContextConfiguration(plant_context_l_,
                                                  gripper_brick_->plant(), q_l);
  const double I = gripper_brick_->brick_frame()
                       .body()
                       .CalcSpatialInertiaInBodyFrame(*plant_context_r_)
                       .CalcRotationalInertia()
                       .get_moments()(0);
  (*y)(2) = I * (v_brick_r_rotation_x - v_brick_l_rotation_x) -
            (total_wrench_r(2) + total_wrench_l(2)) / 2 * dt;
}

void BrickDynamicMidpointIntegrationConstraint::DoEval(
    const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd* y) const {
  DoEvalGeneric<double>(x, y);
}

void BrickDynamicMidpointIntegrationConstraint::DoEval(
    const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd* y) const {
  DoEvalGeneric<AutoDiffXd>(x, y);
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
