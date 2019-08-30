#pragma once

#include "drake/geometry/scene_graph.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/diagram.h"

namespace drake {
namespace examples {
namespace planar_gripper {

template <typename T>
void WeldFingerFrame(multibody::MultibodyPlant<T>* plant) {
  // The finger base link is welded a fixed distance from the world
  // origin, on the Y-Z plane.
  const double kOriginToBaseDistance = 0.19;

  // Before welding, the finger base link sits at the world origin with the
  // finger pointing along the -Z axis, with all joint angles being zero.

  // Weld the finger. Frame F1 corresponds to the base link finger frame.
  math::RigidTransformd X_WF(Eigen::Vector3d::Zero());
  X_WF = X_WF *
         math::RigidTransformd(Eigen::Vector3d(0, 0, kOriginToBaseDistance));
  const multibody::Frame<T>& finger_base_frame =
      plant->GetFrameByName("finger_base");
  plant->WeldFrames(plant->world_frame(), finger_base_frame, X_WF);
}

Eigen::Vector3d GetFingerTipSpherePositionInFingerTip(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph);

double GetFingerTipSphereRadius(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph);

Eigen::Vector3d GetBrickSize(const multibody::MultibodyPlant<double>& plant,
                             const geometry::SceneGraph<double>& scene_graph);

geometry::GeometryId GetFingerTipGeometryId(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph);
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
