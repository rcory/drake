#pragma once

#include "drake/geometry/scene_graph.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/diagram.h"

namespace drake {
namespace examples {
namespace planar_gripper {

template <typename T>
void WeldFingerFrame(multibody::MultibodyPlant<T>* plant, double x_offset = 0);

Eigen::Vector3d GetFingerTipSpherePositionInL2(
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

/// A system that computes the fingertip-sphere contact location in brick frame.
class ContactPointInBrickFrame final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ContactPointInBrickFrame)

  ContactPointInBrickFrame(const multibody::MultibodyPlant<double>& plant);

  void CalcOutput(const systems::Context<double>& context,
                  systems::BasicVector<double> *output) const;

 private:
  const multibody::MultibodyPlant<double>& plant_;
  std::unique_ptr<systems::Context<double>> plant_context_;
};

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
