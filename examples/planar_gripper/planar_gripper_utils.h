#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "drake/examples/planar_gripper/planar_gripper_common.h"

namespace drake {
namespace examples {
namespace planar_gripper {

using Eigen::Vector2d;

struct BrickFaceInfo {
  BrickFaceInfo(const BrickFace face,
                const Eigen::Ref<const Vector2d>& p_BCb,
                bool contact)
      : brick_face(face), p_BCb(p_BCb), is_in_contact(contact) {}
  BrickFace brick_face;   //  the brick face this finger is assigned to.
  Vector2d p_BCb;  // holds the contact or witness point, in Brick frame.

  // A boolean that indicates whether a finger is in contact with this
  // `brick_face`. If true, then this structure stores the actual contact point,
  // given by the contact results object. If false, this structure stores the
  // witness point to a proximity query.
  bool is_in_contact;
};

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
 * Get the MBP body index for the brick.
 */
multibody::BodyIndex GetBrickBodyIndex(
    const multibody::MultibodyPlant<double>& plant);

/**
 * Get the MBP body index for the finger's tip link.
 */
multibody::BodyIndex GetTipLinkBodyIndex(
    const multibody::MultibodyPlant<double>& plant, Finger finger);

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
 * A system that outputs the closest faces to each finger, and the contact (or
 * witness points) on the brick. The output format is std::unordered_map<Finger,
 * BrickFaceInfo>. The input port takes in a geometry::QueryObject<double>
 * object, this input port should be connected to the scene graph geometry query
 * output port.
 */
class FingerFaceAssigner final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FingerFaceAssigner)

  /**
   * FingerFaceAssigner constructor.
   * @param plant The MultiBodyPlant
   * @param scene_graph The SceneGraph
   * @param fingers The Fingers contained in the plant (three by default).
   */
  FingerFaceAssigner(const multibody::MultibodyPlant<double>& plant,
                     const geometry::SceneGraph<double>& scene_graph,
                     const std::vector<Finger>& fingers = {
                         Finger::kFinger1, Finger::kFinger2, Finger::kFinger3});

  const systems::OutputPort<double>& get_finger_face_assignments_output_port()
      const {
    return this->get_output_port(finger_face_assignments_output_port_);
  }

  const systems::InputPort<double>& get_geometry_query_input_port() const {
    return this->get_input_port(geometry_query_input_port_);
  }

 private:
  void CalcFingerFaceAssignments(
      const systems::Context<double>& context,
      std::unordered_map<Finger, BrickFaceInfo>* finger_face_assignments) const;

  const multibody::MultibodyPlant<double>& plant_;
  const geometry::SceneGraph<double>& scene_graph_;
  std::unique_ptr<systems::Context<double>> plant_context_;
  std::unordered_map<Finger, geometry::GeometryId> finger_sphere_geometry_ids_;
  geometry::GeometryId brick_geometry_id_;
  systems::InputPortIndex geometry_query_input_port_{};
  systems::OutputPortIndex finger_face_assignments_output_port_{};
};

/**
 * A system that declares a periodic publish event where the instantaneous
 * positions of the plant (keyframe) is written to standard out.
 * @param plant The MultibodyPlant
 * @param joint_names A vector of joint names that should be included in the
 * keyframe printout.
 * @param do_print_time A boolean indicating whether context time should be printed
 * out.
 */
class PrintKeyframes final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(PrintKeyframes)

  PrintKeyframes(const MultibodyPlant<double>& plant,
                 const std::vector<std::string>& joint_names, double period,
                 bool do_print_time);

 private:
  systems::EventStatus Print(
      const systems::Context<double>& context) const;

  MatrixX<double> Sx_;  // The state selector matrix.
  bool do_print_time_;  // indicates whether to print context time.
};

/// Returns the index in `contact_results` where the fingertip/brick contact
/// information is found. If it isn't found, then
/// index = contact_results.num_point_pair_contacts()
std::optional<int> GetContactPairIndex(
    const multibody::MultibodyPlant<double>& plant,
    const multibody::ContactResults<double>& contact_results,
    const Finger finger);

/// Note: This method is strictly defined for brick only simulation, where
/// spatial forces are applied to the brick directly. Although there are no
/// physical fingers involved in brick only simulation, we enumerate spatial
/// forces with Finger numbers (i.e., as keys in the unordered map) for
/// convenience only.  This method returns an unordered map. It maps a spatial
/// force (i.e. a virtual Finger) to a BrickFaceInfo struct.
std::unordered_map<Finger, BrickFaceInfo> BrickSpatialForceAssignments(
    const std::unordered_set<Finger>& fingers_to_control);

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
