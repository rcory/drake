#pragma once

#include "drake/geometry/scene_graph.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/diagram.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"

namespace drake {
namespace examples {
namespace planar_gripper {
// TODO(rcory) I believe most of the functionality in this file can be moved
//  over to planar_gripper.h/cc
template <typename T>
void WeldFingerFrame(multibody::MultibodyPlant<T>* plant);

Eigen::Vector3d GetFingerTipSpherePositionInLt(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph, const Finger finger);

// TODO(rcory) This method only exists for planar_finger_qp_test. Remove this
//  once I remove the dependency in that test.
double GetFingerTipSphereRadius(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph, Finger finger);

Eigen::Vector3d GetBrickSize(const multibody::MultibodyPlant<double>& plant,
                             const geometry::SceneGraph<double>& scene_graph);

geometry::GeometryId GetFingerTipGeometryId(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph, const Finger finger);

geometry::GeometryId GetBrickGeometryId(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph);

multibody::BodyIndex GetBrickBodyIndex(
    const multibody::MultibodyPlant<double>& plant);

multibody::BodyIndex GetTipLinkBodyIndex(
    const multibody::MultibodyPlant<double>& plant, const Finger finger);

/// A system that computes the fingertip-sphere contact location in brick frame.
class ContactPointInBrickFrame final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ContactPointInBrickFrame)

  ContactPointInBrickFrame(const multibody::MultibodyPlant<double>& plant,
                           const geometry::SceneGraph<double>& scene_graph,
                           const Finger finger = Finger::kFinger1);

  void in_contact(
      const drake::systems::Context<double>& context,
      bool* is_in_contact) const;

  void CalcOutput(const systems::Context<double>& context,
                  systems::BasicVector<double> *output) const;

  const systems::InputPort<double>& get_geometry_query_input_port() const {
    return this->get_input_port(geometry_query_input_port_);
  }

 private:
  const multibody::MultibodyPlant<double>& plant_;
  const geometry::SceneGraph<double>& scene_graph_;
  std::unique_ptr<systems::Context<double>> plant_context_;
  systems::InputPortIndex geometry_query_input_port_{};
  const Finger finger_{Finger::kFinger1};  /* the finger to control */
};

/// A system to convert the individual contact points coming out of
/// ContactPointInBrickFrame into a
/// std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>>{}, where
/// BrickFace would be kClosest (meaning it isn't used) and the Vector2d is the
/// contact point. This is really only needed because the QP controller takes
/// in the unordered_map format.
/// This system declares 3*n input ports, where n is the number of fingers that
/// are under force control.
/// This system declares one output port, that contains the unordered_map
/// (defined above) whose values describe *only fingers that are in contact*.
/// Information for fingers not in contact is not included in the output.
/// Note: If no fingers are in contact the output map will be empty.
// TODO(rcory) Figure out a way to get rid of this translation.
class ContactPointsToFingerFaceAssignments final
    : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ContactPointsToFingerFaceAssignments);

  ContactPointsToFingerFaceAssignments(std::vector<Finger> fingers);

  void CalcOutput(
      const systems::Context<double>& context,
      std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>*
          finger_face_assignments) const;
 private:
  std::vector<Finger> fingers_;
};

/// A system that outputs the force vector portion of ContactResults as
/// well as the vector of plant reaction forces felt at the force sensor weld,
/// both expressed in the world (W) frame.
class ForceDemuxer final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ForceDemuxer)

  ForceDemuxer(const multibody::MultibodyPlant<double>& plant,
               const Finger finger);

  void SetContactResultsForceOutput(const systems::Context<double>& context,
                  systems::BasicVector<double> *output) const;

  void SetReactionForcesOutput(const systems::Context<double>& context,
                               systems::BasicVector<double> *output) const;


  const systems::InputPort<double>& get_contact_results_input_port() const {
    return this->get_input_port(contact_results_input_port_);
  }

  const systems::InputPort<double>& get_reaction_forces_input_port() const {
    return this->get_input_port(reaction_forces_input_port_);
  }

  const systems::InputPort<double>& get_state_input_port() const {
    return this->get_input_port(state_input_port_);
  }

  const systems::OutputPort<double>& get_contact_res_vec_output_port() const {
    return this->get_output_port(contact_results_vec_output_port_);
  }

  const systems::OutputPort<double>& get_reaction_vec_output_port() const {
    return this->get_output_port(reaction_forces_vec_output_port_);
  }

 private:
  const multibody::MultibodyPlant<double> &plant_;
  std::unique_ptr<systems::Context<double>> plant_context_;
  systems::InputPortIndex contact_results_input_port_{};
  systems::InputPortIndex reaction_forces_input_port_{};
  systems::InputPortIndex state_input_port_{};
  systems::OutputPortIndex contact_results_vec_output_port_{};
  systems::OutputPortIndex reaction_forces_vec_output_port_{};
  const Finger finger_;  /* the finger to control */
};

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
