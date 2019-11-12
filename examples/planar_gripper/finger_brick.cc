#include "drake/examples/planar_gripper/finger_brick.h"

namespace drake {
namespace examples {
namespace planar_gripper {

using drake::multibody::ContactResults;

template <typename T>
void WeldFingerFrame(multibody::MultibodyPlant<T>* plant, double z_offset) {
  // The finger base link is welded a fixed distance from the world
  // origin, on the Y-Z plane.
  const double kOriginToBaseDistance = 0.19;

  // Before welding, the finger base link sits at the world origin with the
  // finger pointing along the -Z axis, with all joint angles being zero.

  // Weld the finger. Frame F1 corresponds to the base link finger frame.
  math::RigidTransformd X_WF(Eigen::Vector3d::Zero());
  X_WF = X_WF * math::RigidTransformd(
      Eigen::Vector3d(0, 0, kOriginToBaseDistance + z_offset));
  const multibody::Frame<T>& finger_base_frame =
      plant->GetFrameByName("finger_base");
  plant->WeldFrames(plant->world_frame(), finger_base_frame, X_WF);
}

template void WeldFingerFrame(multibody::MultibodyPlant<double>* plant,
                              double x_offset);

geometry::GeometryId GetBrickGeometryId(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph) {
  const geometry::SceneGraphInspector<double>& inspector =
      scene_graph.model_inspector();
  const geometry::GeometryId brick_geometry_id =
      inspector.GetGeometryIdByName(
          plant.GetBodyFrameIdOrThrow(
              plant.GetBodyByName("brick_base_link").index()),
          geometry::Role::kProximity, "object::box_collision");
  return brick_geometry_id;
}

geometry::GeometryId GetFingerTipGeometryId(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph) {
  const geometry::SceneGraphInspector<double>& inspector =
      scene_graph.model_inspector();
  const geometry::GeometryId finger_tip_geometry_id =
      inspector.GetGeometryIdByName(
          plant.GetBodyFrameIdOrThrow(
              plant.GetBodyByName("finger_tip_link").index()),
          geometry::Role::kProximity, "planar_finger::tip_sphere_collision");
  return finger_tip_geometry_id;
}

Eigen::Vector3d GetFingerTipSpherePositionInLt(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph) {
  const geometry::SceneGraphInspector<double>& inspector =
      scene_graph.model_inspector();
  const geometry::GeometryId finger_tip_geometry_id =
      GetFingerTipGeometryId(plant, scene_graph);
  Eigen::Vector3d p_LtTip =  // position of sphere center in tip-link frame
      inspector.GetPoseInFrame(finger_tip_geometry_id).translation();
  return p_LtTip;
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

/// A utility system for the planar-finger/1-dof brick that extracts the
/// fingertip-sphere/brick contact location in brick frame given from contact
/// results.
/// Note: This contact point doesn't necessarily coincide with the sphere
/// center.
ContactPointInBrickFrame::ContactPointInBrickFrame(
    const multibody::MultibodyPlant<double>& plant, double yc, double zc)
    : plant_(plant),
      plant_context_(plant.CreateDefaultContext()),
      yc_(yc),
      zc_(zc) {
  this->DeclareAbstractInputPort("contact_results",
                                 Value<ContactResults<double>>{});
  this->DeclareVectorInputPort("x",
                               systems::BasicVector<double>(6 /* state */));
  this->DeclareVectorOutputPort("p_BCb",
                                systems::BasicVector<double>(2 /* {y,z} */),
                                &ContactPointInBrickFrame::CalcOutput);
}

void ContactPointInBrickFrame::CalcOutput(
    const drake::systems::Context<double>& context,
    systems::BasicVector<double>* output) const {

  // Get the actual contact force.
  const auto& contact_results =
      this->get_input_port(0).Eval<ContactResults<double>>(context);

  auto state = this->EvalVectorInput(context, 1)->get_value();
  plant_.SetPositions(plant_context_.get(), state.head(3));
  plant_.SetVelocities(plant_context_.get(), state.tail(3));

  const multibody::Frame<double>& brick_frame =
      plant_.GetFrameByName("brick_base_link");

  const multibody::Frame<double>& world_frame =
      plant_.world_frame();

//  drake::log()->info("num contacts: {}", contact_results.num_point_pair_contacts());
//  DRAKE_DEMAND(contact_results.num_point_pair_contacts() == 1);

  auto p_BCb = output->get_mutable_value();

  // The contact point in brick frame. Note the value coming out of contact
  // results gives the contact location in world frame, which we need to convert
  // to brick frame.
  if (contact_results.num_point_pair_contacts() > 0) {
    DRAKE_DEMAND(contact_results.num_point_pair_contacts() == 1);
    Eigen::Vector3d result;
    auto p_WCb =
        contact_results.point_pair_contact_info(0).contact_point();
    plant_.CalcPointsPositions(*plant_context_, world_frame,
                               p_WCb, brick_frame, &result);
    p_BCb = result.tail<2>();
  } else {
    // TODO(rcory) Use the closest point (distance-wise) to the brick instead.
    p_BCb = Eigen::Vector2d(yc_, zc_);
  }


}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
