#include "drake/examples/kuka_iiwa_arm/dev/box_rotation/optitrack_sim.h"
#include "drake/multibody/rigid_body_plant/kinematics_results.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace box_rotation {

using std::pair;
using std::string;
using drake::systems::KinematicsResults;

OptitrackSim::OptitrackSim(const RigidBodyTree<double>& tree) {

  this->DeclareAbstractInputPort(); // of type KinematicsResults
  this->DeclareAbstractOutputPort(&OptitrackSim::MakeOutputStatus,
                                  &OptitrackSim::OutputStatus);

  /// objects (bodies) to track
  /// IDs as defined in Motive
  name_to_id_map_["left_iiwa_link_0"] = 4;
  name_to_id_map_["right_iiwa_link_0"] = 3;
  name_to_id_map_["box"] = 2;

  for (auto it = name_to_id_map_.begin(); it != name_to_id_map_.end(); ++it) {
    id_to_body_index_map_[it->second] = tree.FindBodyIndex(it->first);
  }
}

std::vector<TrackedObject> OptitrackSim::MakeOutputStatus() const {
  std::vector<TrackedObject> mocap_objects(name_to_id_map_.size());
  return mocap_objects;
}

void OptitrackSim::OutputStatus(const systems::Context<double>& context,
                  std::vector<TrackedObject>* output) const {

  std::vector<TrackedObject>& mocap_objects = *output;

  // in here we extract the input port for KinematicsResults object
  // get the transformation of the bodies we care about
  // and fill in the mocap_objects vector for sending out
  const KinematicsResults<double>* kres =
      this->EvalInputValue<KinematicsResults<double>>(context, 0);

  int vind = 0;
  for (auto it = name_to_id_map_.begin(); it != name_to_id_map_.end(); ++it){

    mocap_objects[vind].name = it->first;
    mocap_objects[vind].id = it->second;

    mocap_objects[vind].rot = kres->get_body_orientation(
        id_to_body_index_map_.at(it->second));

    mocap_objects[vind].trans = kres->get_body_position(
        id_to_body_index_map_.at(it->second));

    ++vind;
  }

}

}  // box_rotation
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake