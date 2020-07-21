#pragma once

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/diagram.h"

namespace drake {
namespace examples {
namespace planar_gripper {
// TODO(rcory) I believe most of the functionality in this file can be moved
//  over to planar_gripper.h/cc
template <typename T>
void WeldFingerFrame(multibody::MultibodyPlant<T>* plant, T finger_angle = 0);

Eigen::Vector3d GetFingerTipSpherePositionInLt(
    const multibody::MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph, const Finger finger);

// TODO(rcory) This method only exists for planar_finger_qp_test. Remove this
//  once I remove the dependency in that test.
double GetFingerTipSphereRadius(const multibody::MultibodyPlant<double>& plant,
                                const geometry::SceneGraph<double>& scene_graph,
                                Finger finger);

Eigen::Vector3d GetBrickSize(const multibody::MultibodyPlant<double>& plant,
                             const geometry::SceneGraph<double>& scene_graph);

/// A system that outputs the force vector portion of ContactResults as
/// well as the vector of plant reaction forces felt at the force sensor weld,
/// both expressed in the world (W) frame.
class ForceDemuxer final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ForceDemuxer)

  ForceDemuxer(const multibody::MultibodyPlant<double>& plant,
               const Finger finger);

  void SetContactResultsForceOutput(const systems::Context<double>& context,
                                    systems::BasicVector<double>* output) const;

  void SetReactionForcesOutput(const systems::Context<double>& context,
                               systems::BasicVector<double>* output) const;

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
  const multibody::MultibodyPlant<double>& plant_;
  std::unique_ptr<systems::Context<double>> plant_context_;
  systems::InputPortIndex contact_results_input_port_{};
  systems::InputPortIndex reaction_forces_input_port_{};
  systems::InputPortIndex state_input_port_{};
  systems::OutputPortIndex contact_results_vec_output_port_{};
  systems::OutputPortIndex reaction_forces_vec_output_port_{};
  const Finger finger_; /* the finger to control */
};

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
