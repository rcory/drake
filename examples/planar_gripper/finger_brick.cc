#include "drake/examples/planar_gripper/finger_brick.h"

#include <string>

#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/multibody/tree/weld_joint.h"

namespace drake {
namespace examples {
namespace planar_gripper {

using drake::multibody::ContactResults;

template <typename T>
void WeldFingerFrame(multibody::MultibodyPlant<T>* plant) {
  // The finger base link is welded a fixed distance from the world
  // origin, on the Y-Z plane.
  const double kGripperOriginToBaseDistance = 0.19;
  const double kFinger1Angle = 0;

  // Note: Before welding and with the finger joint angles being zero, the
  // finger base link sits at the world origin with the finger pointing along
  // the world -Z axis.

  // We align the planar gripper coordinate frame G with the world frame W.
  const math::RigidTransformd X_WG = math::RigidTransformd::Identity();

  // Weld the first finger. Finger base links are arranged equidistant along the
  // perimeter of a circle. The first finger is welded kFinger1Angle radians
  // from the +Gz-axis. Frames F1, F2, F3 correspond to the base link finger
  // frames.
  math::RigidTransformd X_GF1 =
      math::RigidTransformd(
          Eigen::AngleAxisd(kFinger1Angle, Eigen::Vector3d::UnitX()),
          Eigen::Vector3d(0, 0, 0)) *
      math::RigidTransformd(
          math::RotationMatrixd(),
          Eigen::Vector3d(0, 0, kGripperOriginToBaseDistance));
  const multibody::Frame<T>& finger1_base_frame =
      plant->GetFrameByName("finger1_base");
  plant->WeldFrames(plant->world_frame(), finger1_base_frame, X_WG * X_GF1);
}

template void WeldFingerFrame(multibody::MultibodyPlant<double>* plant);

geometry::GeometryId GetBrickGeometryId(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph) {
  const geometry::SceneGraphInspector<double>& inspector =
      scene_graph.model_inspector();
  const geometry::GeometryId brick_geometry_id = inspector.GetGeometryIdByName(
      plant.GetBodyFrameIdOrThrow(plant.GetBodyByName("brick_link").index()),
      geometry::Role::kProximity, "brick::box_collision");
  return brick_geometry_id;
}

geometry::GeometryId GetFingerTipGeometryId(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph, const Finger finger) {
  std::string fnum = to_string(finger);
  const geometry::SceneGraphInspector<double>& inspector =
      scene_graph.model_inspector();
  const geometry::GeometryId finger_tip_geometry_id =
      inspector.GetGeometryIdByName(
          plant.GetBodyFrameIdOrThrow(
              plant.GetBodyByName(fnum + "_tip_link").index()),
          geometry::Role::kProximity, "planar_gripper::tip_sphere_collision");
  return finger_tip_geometry_id;
}

Eigen::Vector3d GetFingerTipSpherePositionInLt(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph, const Finger finger) {
  const geometry::SceneGraphInspector<double>& inspector =
      scene_graph.model_inspector();
  const geometry::GeometryId finger_tip_geometry_id =
      GetFingerTipGeometryId(plant, scene_graph, finger);
  Eigen::Vector3d p_LtTip =  // position of sphere center in tip-link frame
      inspector.GetPoseInFrame(finger_tip_geometry_id).translation();
  return p_LtTip;
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

// TODO(rcory) This method only exists for planar_finger_qp_test. Remove this
//  once I remove the dependency in that test.
double GetFingerTipSphereRadius(const multibody::MultibodyPlant<double>& plant,
                                const geometry::SceneGraph<double>& scene_graph,
                                Finger finger) {
  const geometry::SceneGraphInspector<double>& inspector =
      scene_graph.model_inspector();
  const geometry::GeometryId finger_tip_geometry_id =
      GetFingerTipGeometryId(plant, scene_graph, finger);
  const geometry::Shape& fingertip_shape =
      inspector.GetShape(finger_tip_geometry_id);
  double finger_tip_radius =
      dynamic_cast<const geometry::Sphere&>(fingertip_shape).radius();
  return finger_tip_radius;
}

Eigen::Vector3d GetBrickSize(const multibody::MultibodyPlant<double>& plant,
                             const geometry::SceneGraph<double>& scene_graph) {
  const geometry::SceneGraphInspector<double>& inspector =
      scene_graph.model_inspector();
  const geometry::Shape& brick_shape =
      inspector.GetShape(inspector.GetGeometryIdByName(
          plant.GetBodyFrameIdOrThrow(
              plant.GetBodyByName("brick_link").index()),
          geometry::Role::kProximity, "brick::box_collision"));
  const Eigen::Vector3d brick_size =
      dynamic_cast<const geometry::Box&>(brick_shape).size();
  return brick_size;
}

/// Returns the index in `contact_results` where the fingertip/brick contact
/// information is found. If it isn't found, then
/// index = contact_results.num_point_pair_contacts()
int GetContactPairIndex(const multibody::MultibodyPlant<double>& plant,
                        const ContactResults<double>& contact_results,
                        const Finger finger) {
  // Determine whether we have fingertip/brick contact.
  int brick_index = GetBrickBodyIndex(plant);
  int ftip_index = GetTipLinkBodyIndex(plant, finger);
  // find the fingertip/brick contact pair
  int pair_index;
  for (pair_index = 0; pair_index < contact_results.num_point_pair_contacts();
       pair_index++) {
    auto info = contact_results.point_pair_contact_info(pair_index);
    if (info.bodyA_index() == brick_index && info.bodyB_index() == ftip_index) {
      break;
    } else if (info.bodyA_index() == ftip_index &&
               info.bodyB_index() == brick_index) {
      break;
    }
  }
  return pair_index;
}

/// A utility system for the planar-finger/1-dof brick that extracts the
/// fingertip-sphere/brick contact location in brick frame given from contact
/// results.
/// Note: This contact point doesn't necessarily coincide with the sphere
/// center.
ContactPointInBrickFrame::ContactPointInBrickFrame(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph, const Finger finger)
    : plant_(plant),
      scene_graph_(scene_graph),
      plant_context_(plant.CreateDefaultContext()),
      finger_(finger) {
  this->DeclareAbstractInputPort("contact_results",
                                 Value<ContactResults<double>>{});
  this->DeclareVectorInputPort(
      "x", systems::BasicVector<double>(plant.num_positions() +
                                        plant.num_velocities() /* state */));
  this->DeclareVectorOutputPort("p_BrCb",
                                systems::BasicVector<double>(2 /* {y,z} */),
                                &ContactPointInBrickFrame::CalcOutput);

  this->DeclareAbstractOutputPort("b_in_contact",
                                  &ContactPointInBrickFrame::in_contact);

  // This provides the geometry query object, which can compute the witness
  // points between fingertip and box (used in impedance control).
  geometry_query_input_port_ =
      this->DeclareAbstractInputPort("geometry_query",
                                     Value<geometry::QueryObject<double>>{})
          .get_index();
}

void ContactPointInBrickFrame::in_contact(
    const drake::systems::Context<double>& context, bool* is_in_contact) const {
  // Get the actual contact force.
  const auto& contact_results =
      this->get_input_port(0).Eval<ContactResults<double>>(context);

  // TODO(rcory) Replace this with a fingertip force threshold (as it would be
  //  on the real hardware).
  int pair_index = GetContactPairIndex(plant_, contact_results, finger_);
  *is_in_contact = pair_index < contact_results.num_point_pair_contacts();
}

void ContactPointInBrickFrame::CalcOutput(
    const drake::systems::Context<double>& context,
    systems::BasicVector<double>* output) const {
  // Get the actual contact force.
  const auto& contact_results =
      this->get_input_port(0).Eval<ContactResults<double>>(context);

  auto state = this->EvalVectorInput(context, 1)->get_value();
  plant_.SetPositions(plant_context_.get(), state.head(plant_.num_positions()));
  plant_.SetVelocities(plant_context_.get(),
                       state.tail(plant_.num_velocities()));

  // The geometry query object to be used for impedance control (determines
  // closest points between fingertip and brick).
  geometry::QueryObject<double> geometry_query_obj =
      get_geometry_query_input_port().Eval<geometry::QueryObject<double>>(
          context);

  const multibody::Frame<double>& brick_frame =
      plant_.GetFrameByName("brick_link");

  const multibody::Frame<double>& world_frame = plant_.world_frame();

  auto p_BCb = output->get_mutable_value();

  int pair_index = GetContactPairIndex(plant_, contact_results, finger_);

  // The following retrieves the contact point in brick frame. Note the value
  // coming out of contact results gives the contact location in world frame,
  // which we need to convert to brick frame.

  // If we found a fingertip/brick contact then the contact point is given by
  // the contact result's point pair information.
  if (pair_index < contact_results.num_point_pair_contacts()) {
    Eigen::Vector3d result;
    auto p_WCb =
        contact_results.point_pair_contact_info(pair_index).contact_point();
    plant_.CalcPointsPositions(*plant_context_, world_frame, p_WCb, brick_frame,
                               &result);
    p_BCb = result.tail<2>();
  } else {  // Use the closest point (distance-wise) to the brick.
    // First, obtain the closest point on the brick from the fingertip sphere.
    auto pairs_vec =
        geometry_query_obj.ComputeSignedDistancePairwiseClosestPoints();
    DRAKE_DEMAND(pairs_vec.size() >= 1);

    geometry::GeometryId brick_id = GetBrickGeometryId(plant_, scene_graph_);
    geometry::GeometryId ftip_id =
        GetFingerTipGeometryId(plant_, scene_graph_, finger_);

    // Find the pair that contains the brick geometry.
    int pairs_index;
    for (pairs_index = 0; pairs_index < static_cast<int>(pairs_vec.size());
         pairs_index++) {
      if (pairs_vec[pairs_index].id_A == ftip_id &&
          pairs_vec[pairs_index].id_B == brick_id) {
        p_BCb = pairs_vec[pairs_index].p_BCb.tail<2>();
        break;
      } else if (pairs_vec[pairs_index].id_A == brick_id &&
                 pairs_vec[pairs_index].id_B == ftip_id) {
        p_BCb = pairs_vec[pairs_index].p_ACa.tail<2>();
        break;
      }
    }
    if (pairs_index == static_cast<int>(pairs_vec.size())) {
      throw std::runtime_error(
          "Could not find brick box geometry in collision pairs vector.");
    }
  }
}

// TODO(rcory) Remove this method (see header).
ContactPointsToFingerFaceAssignments::ContactPointsToFingerFaceAssignments(
    std::vector<Finger> fingers)
    : fingers_(fingers) {
  for (auto& finger : fingers) {
    if (finger == Finger::kFinger1) {
      this->DeclareVectorInputPort("finger1_contact_point",
                                   systems::BasicVector<double>(2));
      this->DeclareAbstractInputPort("finger1_contact_face",
                                     Value<BrickFace>());
      this->DeclareAbstractInputPort("finger1_b_in_contact", Value<bool>());
    } else if (finger == Finger::kFinger2) {
      this->DeclareVectorInputPort("finger2_contact_point",
                                   systems::BasicVector<double>(2));
      this->DeclareAbstractInputPort("finger2_contact_face",
                                     Value<BrickFace>());
      this->DeclareAbstractInputPort("finger2_b_in_contact", Value<bool>());
    } else if (finger == Finger::kFinger3) {
      this->DeclareVectorInputPort("finger3_contact_point",
                                   systems::BasicVector<double>(2));
      this->DeclareAbstractInputPort("finger3_contact_face",
                                     Value<BrickFace>());
      this->DeclareAbstractInputPort("finger3_b_in_contact", Value<bool>());
    } else {
      throw std::logic_error("Unrecognized Finger.");
    }
  }

  this->DeclareAbstractOutputPort(
      "finger_face_assignments",
      &ContactPointsToFingerFaceAssignments::CalcOutput);
}

void ContactPointsToFingerFaceAssignments::CalcOutput(
    const drake::systems::Context<double>& context,
    std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>*
        finger_face_assignments) const {
  finger_face_assignments->clear();
  for (auto finger : fingers_) {
    bool b_in_contact =
        GetInputPort(to_string(finger) + "_b_in_contact").Eval<bool>(context);
    if (b_in_contact) {
      Eigen::Vector2d finger_contact_pos =
          GetInputPort(to_string(finger) + "_contact_point").Eval(context);
      BrickFace brick_face = GetInputPort(to_string(finger) + "_contact_face")
                                 .Eval<BrickFace>(context);
      finger_face_assignments->emplace(
          finger, std::make_pair(brick_face, finger_contact_pos));
    }
  }
}

ForceDemuxer::ForceDemuxer(const multibody::MultibodyPlant<double>& plant,
                           const Finger finger)
    : plant_(plant), finger_(finger) {
  plant_context_ = plant.CreateDefaultContext();

  contact_results_input_port_ =
      this->DeclareAbstractInputPort("contact_results",
                                     Value<ContactResults<double>>{})
          .get_index();

  reaction_forces_input_port_ =
      this->DeclareAbstractInputPort(
              "reaction_forces",
              Value<std::vector<multibody::SpatialForce<double>>>())
          .get_index();

  contact_results_vec_output_port_ =
      this->DeclareVectorOutputPort("contact_res_forces",
                                    systems::BasicVector<double>(3),
                                    &ForceDemuxer::SetContactResultsForceOutput)
          .get_index();

  reaction_forces_vec_output_port_ =
      this->DeclareVectorOutputPort("reaction_forces",
                                    systems::BasicVector<double>(3),
                                    &ForceDemuxer::SetReactionForcesOutput)
          .get_index();

  state_input_port_ =
      this->DeclareVectorInputPort(
              "x", systems::BasicVector<double>(
                       plant_.num_multibody_states() /* plant state */))
          .get_index();
}

void ForceDemuxer::SetContactResultsForceOutput(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {
  auto output_value = output->get_mutable_value();
  output_value.setZero();

  const auto& contact_results =
      get_contact_results_input_port().Eval<ContactResults<double>>(context);

  // Assume there at most one contact (for now).
  // TODO(rcory) Update this to deal with more than one contact.
  DRAKE_DEMAND(contact_results.num_point_pair_contacts() <= 1);
  if (contact_results.num_point_pair_contacts() > 0) {
    output_value = contact_results.point_pair_contact_info(0).contact_force();
  }
}

void ForceDemuxer::SetReactionForcesOutput(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {
  auto output_value = output->get_mutable_value();
  output_value.setZero();

  auto plant_state =
      this->EvalVectorInput(context, state_input_port_)->get_value();
  plant_.SetPositionsAndVelocities(plant_context_.get(), plant_state);

  const std::vector<multibody::SpatialForce<double>>& spatial_vec =
      get_reaction_forces_input_port()
          .Eval<std::vector<multibody::SpatialForce<double>>>(context);

  std::string fnum = to_string(finger_);
  const multibody::WeldJoint<double>& sensor_joint =
      plant_.GetJointByName<multibody::WeldJoint>(fnum + "_sensor_weldjoint");
  auto sensor_joint_index = sensor_joint.index();

  // Get rotation of child link `tip_link' in the world frame, i.e., R_WC
  const multibody::Body<double>& child = sensor_joint.child_body();
  const math::RigidTransform<double>& X_WC =
      plant_.EvalBodyPoseInWorld(*plant_context_, child);

  output_value =
      X_WC.rotation() * spatial_vec[sensor_joint_index].translational();
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
