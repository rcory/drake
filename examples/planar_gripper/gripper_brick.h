#pragma once

#include "drake/geometry/scene_graph.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/constraint.h"
#include "drake/systems/framework/diagram.h"

namespace drake {
namespace examples {
template <typename T>
class GripperBrickSystem {
 public:
  GripperBrickSystem(bool add_gravity);

  const systems::Diagram<T>& diagram() const { return *diagram_; }

  systems::Diagram<T>* get_mutable_diagram() { return diagram_.get(); }

  const multibody::MultibodyPlant<T>& plant() const { return *plant_; }

  multibody::MultibodyPlant<T>* get_mutable_plant() { return plant_; }

 private:
  std::unique_ptr<systems::Diagram<T>> diagram_;
  multibody::MultibodyPlant<T>* plant_;
  geometry::SceneGraph<T>* scene_graph_;
};

enum class BrickFace {
  kPosZ,
  kNegZ,
  kPosY,
  kNegY,
};

}  // namespace examples
}  // namespace drake
