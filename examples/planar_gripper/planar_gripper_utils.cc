#include "drake/examples/planar_gripper/planar_gripper_utils.h"

namespace drake {
namespace examples {
namespace planar_gripper {
geometry::GeometryId GetFingertipSphereGeometryId(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraphInspector<double>& inspector, Finger finger) {
  return inspector.GetGeometryIdByName(
      plant.GetBodyFrameIdOrThrow(
          plant.GetBodyByName(to_string(finger) + "_tip_link").index()),
      geometry::Role::kProximity, "planar_gripper::tip_sphere_collision");
}

geometry::GeometryId GetBrickGeometryId(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraphInspector<double>& inspector) {
  return inspector.GetGeometryIdByName(
      plant.GetBodyFrameIdOrThrow(plant.GetBodyByName("brick_link").index()),
      geometry::Role::kProximity, "brick::box_collision");
}

namespace {
/**
 * @param query_port This port has to be connected to the scene graph geometry
 * query output port.
 * @param system_context. The context for evaluating geometry query at @p
 * query_port.
 */
std::pair<std::unordered_set<BrickFace>, Eigen::Vector3d>
GetClosestFacesToFingerImpl(const geometry::SceneGraph<double>& scene_graph,
                            geometry::GeometryId finger_sphere_geometry_id,
                            geometry::GeometryId brick_geometry_id,
                            const systems::InputPort<double>& query_port,
                            const systems::Context<double>& system_context) {
  const auto& inspector = scene_graph.model_inspector();

  const auto& query_object =
      query_port.template Eval<geometry::QueryObject<double>>(system_context);
  const geometry::SignedDistancePair<double> signed_distance_pair =
      query_object.ComputeSignedDistancePairClosestPoints(
          finger_sphere_geometry_id, brick_geometry_id);
  const geometry::Box& box_shape =
      dynamic_cast<const geometry::Box&>(inspector.GetShape(brick_geometry_id));
  std::unordered_set<BrickFace> closest_faces;
  const Eigen::Vector3d p_BCb =
      inspector.GetPoseInFrame(brick_geometry_id) * signed_distance_pair.p_BCb;
  if (std::abs(signed_distance_pair.p_BCb(1) - box_shape.depth() / 2) < 1e-3) {
    closest_faces.insert(BrickFace::kPosY);
  } else if (std::abs(signed_distance_pair.p_BCb(1) + box_shape.depth() / 2) <
             1e-3) {
    closest_faces.insert(BrickFace::kNegY);
  }
  if (std::abs(signed_distance_pair.p_BCb(2) - box_shape.height() / 2) < 1e-3) {
    closest_faces.insert(BrickFace::kPosZ);
  } else if (std::abs(signed_distance_pair.p_BCb(2) + box_shape.height() / 2) <
             1e-3) {
    closest_faces.insert(BrickFace::kNegZ);
  }
  return std::make_pair(closest_faces, p_BCb);
}
}  // namespace

std::pair<std::unordered_set<BrickFace>, Eigen::Vector3d>
GetClosestFacesToFinger(const multibody::MultibodyPlant<double>& plant,
                        const geometry::SceneGraph<double>& scene_graph,
                        const systems::Context<double>& plant_context,
                        Finger finger) {
  const auto& inspector = scene_graph.model_inspector();
  const geometry::GeometryId finger_sphere_geometry_id =
      GetFingertipSphereGeometryId(plant, inspector, finger);
  const geometry::GeometryId brick_geometry_id =
      GetBrickGeometryId(plant, inspector);
  const auto& query_port = plant.get_geometry_query_input_port();
  return GetClosestFacesToFingerImpl(scene_graph, finger_sphere_geometry_id,
                                     brick_geometry_id, query_port,
                                     plant_context);
}

FingerFaceAssigner::FingerFaceAssigner(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph)
    : plant_{plant}, scene_graph_{scene_graph} {
  const auto& inspector = scene_graph_.model_inspector();
  for (const auto finger :
       {Finger::kFinger1, Finger::kFinger2, Finger::kFinger3}) {
    finger_sphere_geometry_ids_.emplace(
        finger, GetFingertipSphereGeometryId(plant_, inspector, finger));
  }
  DRAKE_DEMAND(static_cast<int>(finger_sphere_geometry_ids_.size()) ==
               kNumFingers);
  brick_geometry_id_ = GetBrickGeometryId(plant, inspector);
  geometry_query_input_port_ =
      this->DeclareAbstractInputPort("geometry_query",
                                     Value<geometry::QueryObject<double>>{})
          .get_index();
  finger_face_assignments_output_port_ =
      this->DeclareAbstractOutputPort("finger_face_assignments",
                                      &FingerFaceAssigner::CalcOutput)
          .get_index();
}

void FingerFaceAssigner::CalcOutput(
    const systems::Context<double>& context,
    std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>*
        finger_face_assignments) const {
  finger_face_assignments->clear();
  const auto& query_port = this->get_geometry_query_input_port();
  for (const auto& finger_id_pair : finger_sphere_geometry_ids_) {
    const std::pair<std::unordered_set<BrickFace>, Eigen::Vector3d>
        finger_faces = GetClosestFacesToFingerImpl(
            scene_graph_, finger_id_pair.second, brick_geometry_id_, query_port,
            context);
    // There must exist at least one closest face.
    DRAKE_DEMAND(!finger_faces.first.empty());
    // If there are multiple closest faces (when the witness point is a vertex
    // of the brick), we arbitrarily choose the first closest face.
    finger_face_assignments->emplace(
        finger_id_pair.first,
        std::make_pair(*finger_faces.first.begin(),
                       Eigen::Vector2d(finger_faces.second.tail<2>())));
  }
}

PrintKeyframes::PrintKeyframes(const MultibodyPlant<double>& plant,
                               const std::vector<std::string>& joint_names,
                               double period, bool do_print_time)
    : do_print_time_(do_print_time) {
  this->DeclareVectorInputPort(
      "plant_state",
      systems::BasicVector<double>(plant.num_multibody_states()));

  this->DeclarePeriodicPublishEvent(period, 0., &PrintKeyframes::Print);

  Sx_ = MakeStateSelectorMatrix(plant, joint_names);
  std::cout << "keyframe_dt=" << period << std::endl;
  for (const auto& iter : joint_names) {
    std::cout<< iter << " ";
  }
  std::cout << std::endl;
}

systems::EventStatus PrintKeyframes::Print(
    const drake::systems::Context<double>& context) const {
  VectorX<double> state = Sx_ * this->EvalVectorInput(context, 0)->get_value();
  if (do_print_time_) {
    std::cout << context.get_time() << " ";
  }
  std::cout << state.head(state.size() / 2).transpose() << std::endl;

  return systems::EventStatus::Succeeded();
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
