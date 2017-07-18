#include "drake/examples/TRI_Remy/controller/remy_controller.h"
#include "drake/examples/TRI_Remy/remy_common.h"

namespace drake {
namespace examples {
namespace Remy {

using systems::Context;
using systems::BasicVector;

template <typename T>
RemyController<T>::RemyController(
    std::unique_ptr<RigidBodyTree<T>> full_robot)
    : full_robot_(std::move(full_robot)),
      base_link_(full_robot_->FindBody("base_link")) {
  kp_arm_.setConstant(6); kd_arm_.setConstant(6);

  kp_hand_.setConstant(5); kd_hand_.setConstant(5);

  kp_lift_ = 10; kd_lift_ = 5;

  lin_v_gain_ = 10; omega_v_gain_ = 10 * kWheelYOffset;
}

template <typename T>
Vector2<T> RemyController<T>::CalcWheelTorque(const KinematicsCache<T>& cache,
                                               const T v_d, const T w_d) const {
  Vector6<T> V_WB =
      full_robot_->CalcBodySpatialVelocityInWorldFrame(cache, *base_link_);
  Isometry3<T> X_WB = full_robot_->CalcBodyPoseInWorldFrame(cache, *base_link_);

  Vector6<T> V_WB_B;
  V_WB_B.template head<3>() =
      X_WB.linear().transpose() * V_WB.template head<3>();
  V_WB_B.template tail<3>() =
      X_WB.linear().transpose() * V_WB.template tail<3>();

  // xdot in body frame.
  T v = V_WB_B[3];
  // yawdot in body frame.
  T w = V_WB_B[2];

  T right_wheel_trq = lin_v_gain_ * (v_d - v) + omega_v_gain_ * (w_d - w);
  T left_wheel_trq = lin_v_gain_ * (v_d - v) - omega_v_gain_ * (w_d - w);

  return Vector2<T>(right_wheel_trq, left_wheel_trq);
}

template <typename T>
Vector3<T> RemyController<T>::CalcHandTorque(const KinematicsCache<T>& cache,
                                              const Vector3<T>& q_d) const {
  auto q = cache.getQ().template segment<3>(kHandQStart);
  auto v = cache.getV().template segment<3>(kHandVStart);

  Vector3<T> trq =
      (kp_hand_.array() * (q_d - q).array() - kd_hand_.array() * v.array())
          .matrix();

  return trq;
}

template <typename T>
T RemyController<T>::CalcLiftAcc(const KinematicsCache<T>& cache, const T q_d,
                                   const T v_d, const T vd_d) const {
  T q = cache.getQ()[kLiftQStart];
  T v = cache.getV()[kLiftVStart];

  T acc = kp_lift_ * (q_d - q) + kd_lift_ * (v_d - v) + vd_d;
  return acc;
}

template <typename T>
VectorX<T> RemyController<T>::CalcArmAcc(const KinematicsCache<T>& cache,
                                          const VectorX<T>& q_d,
                                          const VectorX<T>& v_d,
                                          const VectorX<T>& vd_d) const {
  auto q = cache.getQ().template segment<kNumArmDofs>(kArmQStart);
  auto v = cache.getV().template segment<kNumArmDofs>(kArmVStart);

  VectorX<T> acc = (kp_arm_.array() * (q_d - q).array() +
      kd_arm_.array() * (v_d - v).array()).matrix() + vd_d;
  return acc;
}

template <typename T>
VectorX<T> RemyController<T>::CalcTorque(const VectorX<T>& acc,
                                          KinematicsCache<T>* cache) const {
  DRAKE_DEMAND(acc.size() == full_robot_->get_num_velocities());
  eigen_aligned_std_unordered_map<RigidBody<T> const*, Vector6<T>> f_ext;

  VectorX<T> full_torque =
      full_robot_->inverseDynamics(*cache, f_ext, acc);

  VectorX<T> mmb_torque = full_torque.segment(kWheelVStart, kNumMMBDofs);
  VectorX<T> arm_torque = full_torque.segment(kArmVStart,
                                              kNumArmDofs+kNumHandDofs);

  VectorX<T> torque(12);
  torque << mmb_torque, arm_torque;

  return torque;
}

template <typename T>
RemyControllerSystem<T>::RemyControllerSystem(
    std::unique_ptr<RigidBodyTree<T>> full_robot)
    : controller_(std::move(full_robot)) {
  const auto& robot = controller_.get_full_robot();

  input_port_index_full_estimated_state_ =
      this->DeclareInputPort(
              systems::kVectorValued,
              robot.get_num_positions() + robot.get_num_velocities())
          .get_index();

  output_port_index_control_ =
      this->DeclareVectorOutputPort(BasicVector<T>(robot.get_num_actuators()),
                                    &RemyControllerSystem<T>::CalcOutputTorque)
          .get_index();
}

template <typename T>
void RemyControllerSystem<T>::CalcOutputTorque(const Context<T>& context,
                                                BasicVector<T>* output) const {
  /*
  output->get_mutable_value().setZero();
  */
  const auto& robot = controller_.get_full_robot();
  // Do kinematics.
  VectorX<T> x = this->EvalEigenVectorInput(
      context, input_port_index_full_estimated_state_);
  KinematicsCache<T> cache = robot.CreateKinematicsCache();
  cache.initialize(x.head(robot.get_num_positions()),
                   x.tail(robot.get_num_velocities()));
  robot.doKinematics(cache, true);

  VectorX<T> vd_d = VectorX<T>::Zero(robot.get_num_velocities());
  // Lift acc
  vd_d[kLiftVStart] = controller_.CalcLiftAcc(cache, 0.1, 0, 0);

  // Arms acc
  VectorX<T> arm_q = VectorX<T>::Zero(kNumArmDofs);
  arm_q << 3, -1.85, -0.8, 3.14;

  vd_d.template segment<kNumArmDofs>(kArmVStart) = controller_.CalcArmAcc(
      cache, arm_q, VectorX<T>::Zero(kNumArmDofs),
      VectorX<T>::Zero(kNumArmDofs));

  // Call ID.
  output->get_mutable_value() = controller_.CalcTorque(vd_d, &cache);

  // Wheels
  output->get_mutable_value().template head<2>() =
      controller_.CalcWheelTorque(cache, 0.05, 0.1);

  // Hand
  output->get_mutable_value().template tail<kNumHandDofs>() =
      controller_.CalcHandTorque(cache, Vector3<T>::Ones());

}

template class RemyController<double>;
template class RemyControllerSystem<double>;

}  // namespace Remy
}  // namespace examples
}  // namespace drake
