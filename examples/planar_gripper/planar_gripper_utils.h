#pragma once

#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "drake/examples/planar_gripper/planar_gripper_common.h"

namespace drake {
namespace examples {
namespace planar_gripper {
/**
 * Get the geometry ID of the sphere on the finger tip.
 */
geometry::GeometryId GetFingertipSphereGeometryId(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraphInspector<double>& inspector, Finger finger);

/**
 * Get the geometry ID of the brick.
 */
geometry::GeometryId GetBrickGeometryId(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraphInspector<double>& inspector);

/**
 * Compute the closest face(s) to a center finger given the posture.
 * @param plant The multibody plant that contains both the gripper and the
 * brick.
 * @param scene_graph The SceneGraph object registers @p plant as the geometry
 * source.
 * @param plant_context The context of @p plant. Note that this plant_context
 * has to be obtained from the diagram_context, and the diagram_context is the
 * context for both @p plant and @p scene_graph. One way to obtain this context
 * is as follows
 * @code{.cc}
 * DiagramBuilder builder;
 * std::tie(plant, scene_graph) = AddMultibodyPlantSceneGraph(&builder, 0.);
 * auto diagram = builder.Build();
 * auto diagram_context = builder.CreateDefaultContext();
 * auto plant_context = diagram.GetSubsystemContect(*plant, diagram_context);
 * @endcode
 * @param finger The finger to which the closest faces are queried.
 * @return (closest_faces, p_BCb) When the witness point Cb on the brick is at
 * the vertex of the brick, then we return the two neighbouring faces of that
 * vertex. Otherwise we return the unique face on which the witness point Cb
 * lives. p_BCb is a 3 x 1 position vector, it is the position of the brick
 * witness point Cb on the brick frame B.
 */
std::pair<std::unordered_set<BrickFace>, Eigen::Vector3d>
GetClosestFacesToFinger(const multibody::MultibodyPlant<double>& plant,
                        const geometry::SceneGraph<double>& scene_graph,
                        const systems::Context<double>& plant_context,
                        Finger finger);

/**
 * A system that outputs the closest faces to each finger, and the witness
 * points on the brick. The output format is std::unordered_map<Finger,
 * std::pair<BrickFace, Eigen::Vector2d>>. The input port takes in a
 * geometry::QueryObject<double> object, this input port should be connected to
 * the scene graph geometry query output port.
 */
class FingerFaceAssigner final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FingerFaceAssigner)

  FingerFaceAssigner(const multibody::MultibodyPlant<double>& plant,
                     const geometry::SceneGraph<double>& scene_graph);

  const systems::OutputPort<double>& get_finger_face_assignments_output_port()
      const {
    return this->get_output_port(finger_face_assignments_output_port_);
  }

  const systems::InputPort<double>& get_geometry_query_input_port() const {
    return this->get_input_port(geometry_query_input_port_);
  }

 private:
  void CalcOutput(
      const systems::Context<double>& context,
      std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>*
          finger_face_assignments) const;

  const multibody::MultibodyPlant<double>& plant_;
  const geometry::SceneGraph<double>& scene_graph_;
  std::unordered_map<Finger, geometry::GeometryId> finger_sphere_geometry_ids_;
  geometry::GeometryId brick_geometry_id_;
  systems::InputPortIndex geometry_query_input_port_{};
  systems::OutputPortIndex finger_face_assignments_output_port_{};
};

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
