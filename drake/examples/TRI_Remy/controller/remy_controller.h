#pragma once

#include <memory>
#include <string>

#include "drake/common/drake_copyable.h"
#include "drake/multibody/rigid_body_tree.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace examples {
namespace Remy {

// The number of actuated dofs for various parts of the robot
constexpr int kNumMMBDofs = 3;
constexpr int kNumArmDofs = 6;
constexpr int kNumHandDofs = 3; // only includes fingers

template <typename T>
class RemyController {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(RemyController)

  explicit RemyController(std::unique_ptr<RigidBodyTree<T>> full_robot);

  Vector2<T> CalcWheelTorque(const KinematicsCache<T>& cache, const T v_d,
                             const T w_d) const;

  Vector3<T> CalcHandTorque(const KinematicsCache<T>& cache,
                            const Vector3<T>& q_d) const;

  T CalcLiftAcc(const KinematicsCache<T>& cache, const T q_d, const T v_d,
                 const T vd_d = 0) const;

  VectorX<T> CalcArmAcc(const KinematicsCache<T>& cache, const VectorX<T>& q_d,
                        const VectorX<T>& v_d, const VectorX<T>& vd_d) const;

  VectorX<T> CalcTorque(const VectorX<T>& acc, KinematicsCache<T>* cache) const;

  const RigidBodyTree<T>& get_full_robot() const { return *full_robot_; }

 private:
  std::unique_ptr<RigidBodyTree<T>> full_robot_{nullptr};
  const RigidBody<T>* const base_link_{nullptr};

  // The magic number is wheel to center.
  static constexpr double kWheelYOffset = 0.24193;

  VectorX<T> kp_arm_ = VectorX<T>::Zero(kNumArmDofs);
  VectorX<T> kd_arm_ = VectorX<T>::Zero(kNumArmDofs);

  T kp_lift_;
  T kd_lift_;

  T lin_v_gain_;
  T omega_v_gain_;

  VectorX<T> kp_hand_ = VectorX<T>::Zero(kNumHandDofs);
  VectorX<T> kd_hand_ = VectorX<T>::Zero(kNumHandDofs);
};

template <typename T>
class RemyControllerSystem : public systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(RemyControllerSystem)

  explicit RemyControllerSystem(std::unique_ptr<RigidBodyTree<T>> full_robot);

  const systems::InputPortDescriptor<T>& get_input_port_full_estimated_state()
  const {
    return this->get_input_port(input_port_index_full_estimated_state_);
  }

  /**
   * Returns the output port for computed control.
   */
  const systems::OutputPort<T>& get_output_port_control() const {
    return this->get_output_port(output_port_index_control_);
  }

  /**
   * Returns a constant reference to the RigidBodyTree used for control.
   */
  const RigidBodyTree<T>& get_full_robot() const {
    return controller_.get_full_robot();
  }

 private:
  void CalcOutputTorque(const systems::Context<T>& context,
                        systems::BasicVector<T>* output) const;

  RemyController<T> controller_;

  int input_port_index_full_estimated_state_{-1};

  int output_port_index_control_{-1};
};

}  // namespace Remy
}  // namespace examples
}  // namespace drake
