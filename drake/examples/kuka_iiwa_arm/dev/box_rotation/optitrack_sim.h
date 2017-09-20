#pragma once

#include <vector>

#include "drake/systems/framework/leaf_system.h"
#include "drake/multibody/rigid_body_tree.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace box_rotation {

struct TrackedObject {
  int id; // optitrack id
  std::string name;
  Eigen::Quaterniond rot;
  Eigen::Vector3d trans;
};


class OptitrackSim : public systems::LeafSystem<double> {
 public:
  explicit OptitrackSim(const RigidBodyTree<double>& tree);

  const systems::InputPortDescriptor<double>& get_kinematics_input_port() const {
    return this->get_input_port(0);
  }

  const systems::OutputPort<double>& get_pose_output_port() const {
    return this->get_output_port(0);
  }

 private:
  std::vector<TrackedObject> MakeOutputStatus() const;

  void OutputStatus(const systems::Context<double>& context,
                    std::vector<TrackedObject>* output) const;

  std::map<std::string, int> name_to_id_map_; // maps urdf link name to Optitrack id
  std::map<int, int> id_to_body_index_map_; // maps Optitrack id to body index in the RBT
};


}  // box_rotation
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake