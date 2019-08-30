#include "drake/examples/planar_gripper/finger_brick.h"

namespace drake {
namespace examples {
namespace planar_gripper {
geometry::GeometryId GetFingerTipGeometryId(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph) {
  const geometry::SceneGraphInspector<double>& inspector =
      scene_graph.model_inspector();
  const geometry::GeometryId finger_tip_geometry_id =
      inspector.GetGeometryIdByName(
          plant.GetBodyFrameIdOrThrow(
              plant.GetBodyByName("finger_link2").index()),
          geometry::Role::kProximity, "planar_gripper::link2_pad_collision");
  return finger_tip_geometry_id;
}

Eigen::Vector3d GetFingerTipSpherePositionInFingerTip(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph) {
  const geometry::SceneGraphInspector<double>& inspector =
      scene_graph.model_inspector();
  const geometry::GeometryId finger_tip_geometry_id =
      GetFingerTipGeometryId(plant, scene_graph);
  Eigen::Vector3d p_L2Tip =
      inspector.GetPoseInFrame(finger_tip_geometry_id).translation();
  return p_L2Tip;
}

double GetFingerTipSphereRadius(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph) {
  const geometry::SceneGraphInspector<double>& inspector =
      scene_graph.model_inspector();
  const geometry::GeometryId finger_tip_geometry_id =
      GetFingerTipGeometryId(plant, scene_graph);
  const geometry::Shape& fingertip_shape =
      inspector.GetShape(finger_tip_geometry_id);
  double finger_tip_radius =
      dynamic_cast<const geometry::Sphere&>(fingertip_shape).get_radius();
  return finger_tip_radius;
}

Eigen::Vector3d GetBrickSize(const multibody::MultibodyPlant<double>& plant,
                             const geometry::SceneGraph<double>& scene_graph) {
  const geometry::SceneGraphInspector<double>& inspector =
      scene_graph.model_inspector();
  const geometry::Shape& brick_shape =
      inspector.GetShape(inspector.GetGeometryIdByName(
          plant.GetBodyFrameIdOrThrow(
              plant.GetBodyByName("brick_base_link").index()),
          geometry::Role::kProximity, "object::box_collision"));
  const Eigen::Vector3d brick_size =
      dynamic_cast<const geometry::Box&>(brick_shape).size();
  return brick_size;
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
