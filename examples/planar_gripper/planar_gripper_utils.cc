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

multibody::BodyIndex GetBrickBodyIndex(
    const multibody::MultibodyPlant<double>& plant) {
  return plant.GetBodyByName("brick_link").index();
}

multibody::BodyIndex GetTipLinkBodyIndex(
    const multibody::MultibodyPlant<double>& plant, const Finger finger) {
  std::string fnum = to_string(finger);
  return plant.GetBodyByName(fnum + "_tip_link").index();
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
    const geometry::SceneGraph<double>& scene_graph,
    const std::vector<Finger>& fingers)
    : plant_{plant},
      scene_graph_{scene_graph},
      plant_context_(plant.CreateDefaultContext()) {
  const auto& inspector = scene_graph_.model_inspector();
  for (const auto finger : fingers) {
    finger_sphere_geometry_ids_.emplace(
        finger, GetFingertipSphereGeometryId(plant_, inspector, finger));
  }
  DRAKE_DEMAND(fingers.size() <= kNumFingers);
  brick_geometry_id_ = GetBrickGeometryId(plant, inspector);
  geometry_query_input_port_ =
      this->DeclareAbstractInputPort("geometry_query",
                                     Value<geometry::QueryObject<double>>{})
          .get_index();
  this->DeclareAbstractInputPort("contact_results",
                                 Value<multibody::ContactResults<double>>{});

  this->DeclareVectorInputPort(
      "plant_state",
      systems::BasicVector<double>(plant_.num_multibody_states()));

  finger_face_assignments_output_port_ =
      this->DeclareAbstractOutputPort(
              "finger_face_assignments",
              &FingerFaceAssigner::CalcFingerFaceAssignments)
          .get_index();
}

void FingerFaceAssigner::CalcFingerFaceAssignments(
    const systems::Context<double>& context,
    std::unordered_map<Finger, BrickFaceInfo>* finger_face_assignments) const {
  finger_face_assignments->clear();
  const auto& query_port = this->get_geometry_query_input_port();
  for (const auto& finger_id_pair : finger_sphere_geometry_ids_) {
    const std::pair<std::unordered_set<BrickFace>, Eigen::Vector3d>
        brick_faces = GetClosestFacesToFingerImpl(
            scene_graph_, finger_id_pair.second, brick_geometry_id_, query_port,
            context);
    // There must exist at least one closest face.
    DRAKE_DEMAND(!brick_faces.first.empty());
    // If there are multiple closest faces (when the witness point is a vertex
    // of the brick), we arbitrarily choose the first closest face.

    // Determine whether this finger is in contact or not.
    const auto& contact_results =
        this->GetInputPort("contact_results")
            .Eval<multibody::ContactResults<double>>(context);

    // TODO(rcory) Determine contact using the fingertip force (as it would be
    //  on the real hardware).
    std::optional<int> pair_index =
        GetContactPairIndex(plant_, contact_results, finger_id_pair.first);
    bool is_in_contact = pair_index.has_value();

    // If the geometries are in contact, return the actual contact point.
    // Otherwise, return the witness point. Both are expressed in brick frame.
    if (is_in_contact) {
      auto plant_state =
          this->EvalVectorInput(context,
                                GetInputPort("plant_state").get_index())
              ->get_value();
      plant_.SetPositions(plant_context_.get(),
                          plant_state.head(plant_.num_positions()));
      plant_.SetVelocities(plant_context_.get(),
                           plant_state.tail(plant_.num_velocities()));
      auto p_WCb = contact_results.point_pair_contact_info(pair_index.value())
                       .contact_point();
      Eigen::Vector3d result;
      plant_.CalcPointsPositions(*plant_context_, plant_.world_frame(), p_WCb,
                                 plant_.GetFrameByName("brick_link"),
                                 &result);
      BrickFaceInfo face_info(*brick_faces.first.begin(),
                              Eigen::Vector2d(result.tail<2>()),
                              true /* in contact */);
      finger_face_assignments->emplace(finger_id_pair.first, face_info);
    } else {
      BrickFaceInfo face_info(*brick_faces.first.begin(),
                              Eigen::Vector2d(brick_faces.second.tail<2>()),
                              false /* no contact */);
      finger_face_assignments->emplace(finger_id_pair.first, face_info);
    }
  }
}

/// Returns a std::optional int index in `contact_results` where the
/// fingertip/brick contact information is found, if it exists. If it isn't
/// found, then returns an emtpy std::optinal.
std::optional<int> GetContactPairIndex(
    const multibody::MultibodyPlant<double>& plant,
    const multibody::ContactResults<double>& contact_results,
    const Finger finger) {
  // Determine whether we have fingertip/brick contact.
  int brick_index = GetBrickBodyIndex(plant);
  int ftip_index = GetTipLinkBodyIndex(plant, finger);
  std::optional<int> pair_index;

  for (int index = 0; index < contact_results.num_point_pair_contacts();
       index++) {
    auto& info = contact_results.point_pair_contact_info(index);
    if ((info.bodyA_index() == brick_index &&
         info.bodyB_index() == ftip_index) ||
        (info.bodyA_index() == ftip_index &&
         info.bodyB_index() == brick_index)) {
      pair_index.emplace(index);
      break;
    }
  }
  return pair_index;
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

std::unordered_map<Finger, BrickFaceInfo> BrickSpatialForceAssignments(
    const std::unordered_set<Finger>& fingers_to_control) {
  std::unordered_map<Finger, BrickFaceInfo> brick_spatial_force_assignments;
  // Iterate over virtual fingers (i.e., spatial forces) for brick only sim.
  const double kBoxDimension = 0.1;
  for (const auto& finger : fingers_to_control) {
    if (finger == Finger::kFinger1) {
      brick_spatial_force_assignments.emplace(
          finger, BrickFaceInfo(BrickFace::kNegY,
                                Eigen::Vector2d(-kBoxDimension / 2, 0),
                                true /* is in contact */));
    }
    if (finger == Finger::kFinger2) {
      brick_spatial_force_assignments.emplace(
          finger, BrickFaceInfo(BrickFace::kPosY,
                                Eigen::Vector2d(kBoxDimension / 2, 0),
                                true /* is in contact */));
    }
    if (finger == Finger::kFinger3) {
      brick_spatial_force_assignments.emplace(
          finger, BrickFaceInfo(BrickFace::kNegZ,
                                Eigen::Vector2d(0, -kBoxDimension / 2),
                                true /* is in contact */));
    }
  }
  return brick_spatial_force_assignments;
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
