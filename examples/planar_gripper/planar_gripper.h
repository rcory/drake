#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace examples {
namespace planar_gripper {

using multibody::ModelInstanceIndex;

class PlanarGripper : public systems::Diagram<double> {
 public:
  explicit PlanarGripper(double time_step = 1e-3,
                         bool use_position_control = true);

  /// Sets up the diagram using the planar brick.
  // TODO(rcory) Rename this to something like
  //   SetupPlanarBrickPlantAndFinalize()
  void SetupPlanarBrick(std::string orientation);

  // Sets up the 1-dof pin brick.
  void SetupPinBrick(std::string orientation);

  // TODO(rcory) Rename this to something like FinalizeAndBuild()
  void Finalize();

  /// Returns a reference to the main plant responsible for the dynamics of
  /// the robot and the environment.  This can be used to, e.g., add
  /// additional elements into the world before calling Finalize().
  const multibody::MultibodyPlant<double>& get_multibody_plant() const {
    return *plant_;
  }

  /// Returns a mutable reference to the main plant responsible for the
  /// dynamics of the robot and the environment.  This can be used to, e.g.,
  /// add additional elements into the world before calling Finalize().
  multibody::MultibodyPlant<double>& get_mutable_multibody_plant() {
    return *plant_;
  }

  /// Returns a reference to the SceneGraph responsible for all of the geometry
  /// for the robot and the environment.  This can be used to, e.g., add
  /// additional elements into the world before calling Finalize().
  const geometry::SceneGraph<double>& get_scene_graph() const {
    return *scene_graph_;
  }

  /// Returns a mutable reference to the SceneGraph responsible for all of the
  /// geometry for the robot and the environment.  This can be used to, e.g.,
  /// add additional elements into the world before calling Finalize().
  geometry::SceneGraph<double>& get_mutable_scene_graph() {
    return *scene_graph_;
  }

  /// Return a reference to the plant used by the inverse dynamics controller
  /// (which contains only a model of the gripper).
  const multibody::MultibodyPlant<double>& get_control_plant() const {
    return *owned_control_plant_;
  }

  /// Get the number of joints in the gripper (only -- does not include the
  /// brick).
  int num_gripper_joints() const { return kNumGripperJoints; }

  /// Creates a position vector (in the simulation MBP joint position index
  /// ordering) from the named joints and values in `map_in`.
  VectorX<double> MakeGripperPositionVector(
      const std::map<std::string, double>& map_in);

  /// Convenience method for getting all of the joint angles of the gripper.
  /// This does not include the brick.
  VectorX<double> GetGripperPosition(
      const systems::Context<double>& diagram_context) const;

  /// Convenience method for setting all of the joint angles of the gripper.
  /// @p q must have size num_gripper_joints().
  /// @pre `state` must be the systems::State<double> object contained in
  /// `diagram_context`.
  void SetGripperPosition(const systems::Context<double>& diagram_context,
                          systems::State<double>* diagram_state,
                          const Eigen::Ref<const VectorX<double>>& q) const;

  /// Convenience method for setting all of the joint angles of gripper.
  /// @p q must have size num_gripper_joints().
  void SetGripperPosition(systems::Context<double>* diagram_context,
                          const Eigen::Ref<const VectorX<double>>& q) const {
    SetGripperPosition(*diagram_context, &diagram_context->get_mutable_state(),
                       q);
  }

  /// Convenience method for getting all of the joint velocities of the gripper.
  VectorX<double> GetGripperVelocity(
      const systems::Context<double>& diagram_context) const;

  /// Convenience method for setting all of the joint velocities of the Gripper.
  /// @v must have size num_gripper_joints().
  /// @pre `state` must be the systems::State<double> object contained in
  /// `diagram_context`.
  void SetGripperVelocity(const systems::Context<double>& diagram_context,
                          systems::State<double>* diagram_state,
                          const Eigen::Ref<const VectorX<double>>& v) const;

  /// Convenience method for setting all of the joint velocities of the gripper.
  /// @v must have size num_gripper_joints().
  void SetGripperVelocity(systems::Context<double>* diagram_context,
                          const Eigen::Ref<const VectorX<double>>& v) const {
    SetGripperVelocity(*diagram_context, &diagram_context->get_mutable_state(),
                       v);
  }

  /// Creates a position vector (in MBP joint position index ordering)
  /// from the named joints and values in `map_in`.
  VectorX<double> MakeBrickPositionVector(
      const std::map<std::string, double>& map_in);

  /// Utility function for creating position vectors.
  VectorX<double> MakePositionVector(
      const std::map<std::string, double>& map_in,
      const int num_positions) const;

  /// Creates a velocity vector (in MBP joint velocity index ordering)
  /// from the named joints and values in `map_in`.
  VectorX<double> MakeBrickVelocityVector(
      const std::map<std::string, double>& map_in);

  /// Utility function for creating position vectors.
  VectorX<double> MakeVelocityVector(
      const std::map<std::string, double>& map_in,
      const int num_velocities) const;

  /// Convenience method for setting all of the joint angles of the brick.
  /// @p q must have size 3 (y, z, theta).
  // TODO(rcory) Implement the const Context version that sets State instead.
  void SetBrickPosition(const systems::Context<double>& diagram_context,
                        systems::State<double>* diagram_state,
                        const Eigen::Ref<const VectorX<double>>& q) const;

  /// Convenience method for setting all of the joint angles of brick.
  /// @p q must have size 3 (y, z, theta).
  void SetBrickPosition(systems::Context<double>* diagram_context,
                        const Eigen::Ref<const VectorX<double>>& q) const {
    SetBrickPosition(*diagram_context, &diagram_context->get_mutable_state(),
                     q);
  }

  /// Convenience method for setting all of the joint velocities for the Brick.
  /// @v must have size num_brick_joints().
  /// @pre `state` must be the systems::State<double> object contained in
  /// `diagram_context`.
  void SetBrickVelocity(const systems::Context<double>& diagram_context,
                        systems::State<double>* diagram_state,
                        const Eigen::Ref<const VectorX<double>>& v) const;

  /// Convenience method for setting all of the joint velocities for the Brick.
  /// @v must have size num_brick_joints().
  void SetBrickVelocity(systems::Context<double>* diagram_context,
                        const Eigen::Ref<const VectorX<double>>& v) const {
    SetBrickVelocity(*diagram_context, &diagram_context->get_mutable_state(),
                     v);
  }

  void AddFloor(MultibodyPlant<double>* plant,
                const geometry::SceneGraph<double>& scene_graph);

  void set_brick_floor_penetration(double value) {
    if (is_plant_finalized_) {
      throw std::logic_error(
          "set_brick_floor_penetration must be called before "
          "SetupPlanarBrick() or SetupPinBrick().");
    }
    brick_floor_penetration_ = value;
  }

  void set_floor_coef_static_friction(double value) {
    if (is_plant_finalized_) {
      throw std::logic_error(
          "set_floor_coef_static_friction must be called before "
          "SetupPlanarBrick() or SetupPinBrick().");
    }
    floor_coef_static_friction_ = value;
  }

  void set_floor_coef_kinetic_friction(double value) {
    if (is_plant_finalized_) {
      throw std::logic_error(
          "set_floor_coef_kinetic_friction must be called before "
          "SetupPlanarBrick() or SetupPinBrick().");
    }
    floor_coef_kinetic_friction_ = value;
  }

  void set_penetration_allowance(double value) {
    if (!is_plant_finalized_) {
      throw std::logic_error(
          "set_penetration_allowance must be called after "
          "SetupPlanarBrick() or SetupPinBrick().");
    }
    plant_->set_penetration_allowance(value);
  }

  void set_stiction_tolerance(double value) {
    if (!is_plant_finalized_) {
      throw std::logic_error(
          "set_stiction_tolerance must be called after "
          "SetupPlanarBrick() or SetupPinBrick().");
    }
    plant_->set_stiction_tolerance(value);
  }

  void set_X_WG(math::RigidTransformd X_WG) { X_WG_ = X_WG; }

  /// Sets gravity to zero in the MBP.
  /// @pre Must be called after configuring the MBP by calling either
  ///      SetupPlanarBrick() or SetupPinBrick().
  /// @pre Must be called before finalizing the diagram via Finalize().
  void zero_gravity() {
    if (!is_plant_finalized_) {
      throw std::logic_error(
          "zero_gravity() must be called after SetupPlanarBrick() or "
          "SetupPinBrick().");
    }
    if (is_diagram_finalized_) {
      throw std::logic_error(
          "zero_gravity() must be called before Finalize().");
    }
    // Set the gravity field to zero.
    plant_->mutable_gravity_field().set_gravity_vector(Vector3d::Zero());
    control_plant_->mutable_gravity_field().set_gravity_vector(
        Vector3d::Zero());
  }

  ModelInstanceIndex get_brick_index() const { return brick_index_; }

  ModelInstanceIndex get_planar_gripper_index() const { return gripper_index_; }

  int get_num_gripper_positions() const {
    return plant_->num_positions(gripper_index_);
  }

  int get_num_gripper_velocities() const {
    return plant_->num_velocities(gripper_index_);
  }

  int get_num_gripper_states() const {
    return plant_->num_multibody_states(gripper_index_);
  }

  int get_num_brick_positions() const {
    return plant_->num_positions(brick_index_);
  }

  int get_num_brick_velocities() const {
    return plant_->num_velocities(brick_index_);
  }

  int get_num_brick_states() const {
    return plant_->num_multibody_states(brick_index_);
  }

  double GetBrickPinJointDamping() const;
  Vector3d GetBrickMoments() const;
  double GetBrickMass() const;

  /**
   * Get the geometry id of the finger tip sphere in the plant.
   */
  geometry::GeometryId fingertip_sphere_geometry_id(Finger finger) const;

  /**
   * Get the geometry id of the brick in the plant.
   */
  geometry::GeometryId brick_geometry_id() const;

 private:
  void SetupPlant(std::string orientation, std::string brick_file_name);

  // These are only valid until Finalize() is called.
  std::unique_ptr<multibody::MultibodyPlant<double>> owned_plant_;
  std::unique_ptr<geometry::SceneGraph<double>> owned_scene_graph_;

  // These are valid for the lifetime of this system.
  std::unique_ptr<multibody::MultibodyPlant<double>> owned_control_plant_;
  multibody::MultibodyPlant<double>* control_plant_;
  multibody::MultibodyPlant<double>* plant_;
  geometry::SceneGraph<double>* scene_graph_;
  bool is_plant_finalized_{false};
  bool is_diagram_finalized_{false};

  ModelInstanceIndex gripper_index_;
  ModelInstanceIndex brick_index_;

  bool use_position_control_{true};
  double brick_floor_penetration_{0};  // For the vertical case.
  double floor_coef_static_friction_{0};
  double floor_coef_kinetic_friction_{0};

  // The planar gripper frame G's transform w.r.t. the world frame W.
  math::RigidTransformd X_WG_;

  std::unordered_map<Finger, geometry::GeometryId>
      fingertip_sphere_geometry_ids_;
  geometry::GeometryId brick_geometry_id_;
};

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

