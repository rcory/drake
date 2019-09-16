#pragma once

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/multibody/tree/multibody_tree.h"

namespace drake {
namespace multibody {

template <class T>
struct SpatialForceOutput {
  SpatialForceOutput(
    const Vector3<T>& point_W, const SpatialForce<T>& Force_p_W) :
      p_W(point_W), F_p_W(Force_p_W) { }

  /// Point of origination of the arrow for visualizing the force vector, where
  /// the point represents a vector expressed in the world frame. This point
  /// should generally be coincident with the center-of-mass of a body. If this
  /// point is not coincident with the CoM, the moment will not properly
  /// represent the point about which the body will accelerate rotationally.
  Vector3<T> p_W;

  /// Scaled spatial force applied at point p and expressed in the world frame.
  /// Scaling might be required to properly visualize the force arrow.
  SpatialForce<T> F_p_W;
};

}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class drake::multibody::SpatialForceOutput)
