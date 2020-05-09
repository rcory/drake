#include "drake/examples/planar_gripper/finger_brick.h"

#include <string>

#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/examples/planar_gripper/planar_gripper_utils.h"
#include "drake/multibody/tree/weld_joint.h"

namespace drake {
namespace examples {
namespace planar_gripper {

using drake::multibody::ContactResults;

template <typename T>
void WeldFingerFrame(multibody::MultibodyPlant<T>* plant, T finger_angle) {
  // The finger base link is welded a fixed distance from the world
  // origin, on the Y-Z plane.
  const double kGripperOriginToBaseDistance = 0.19;
  const double kFinger1Angle = finger_angle;

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

template void WeldFingerFrame(multibody::MultibodyPlant<double>* plant,
                              double finger_angle);

Eigen::Vector3d GetFingerTipSpherePositionInLt(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph, const Finger finger) {
  const geometry::SceneGraphInspector<double>& inspector =
      scene_graph.model_inspector();
  const geometry::GeometryId finger_tip_geometry_id =
      GetFingertipSphereGeometryId(plant, scene_graph.model_inspector(),
                                   finger);
  Eigen::Vector3d p_LtTip =  // position of sphere center in tip-link frame
      inspector.GetPoseInFrame(finger_tip_geometry_id).translation();
  return p_LtTip;
}

// TODO(rcory) This method only exists for planar_finger_qp_test. Remove this
//  once I remove the dependency in that test.
double GetFingerTipSphereRadius(const multibody::MultibodyPlant<double>& plant,
                                const geometry::SceneGraph<double>& scene_graph,
                                Finger finger) {
  const geometry::SceneGraphInspector<double>& inspector =
      scene_graph.model_inspector();
  const geometry::GeometryId finger_tip_geometry_id =
      GetFingertipSphereGeometryId(plant, scene_graph.model_inspector(),
                                   finger);
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
