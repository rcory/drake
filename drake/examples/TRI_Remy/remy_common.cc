#include "drake/examples/TRI_Remy/remy_common.h"

#include "drake/multibody/parsers/urdf_parser.h"

using Eigen::Vector3d;
using Eigen::VectorXd;

namespace drake {
namespace examples {
namespace Remy {

// See the @file docblock in remy_common.h for joint index descriptions.
void VerifyRemyTree(const RigidBodyTree<double>& tree) {

  std::map<std::string, int> name_to_idx = tree.computePositionNameToIndexMap();

  DRAKE_DEMAND(name_to_idx.count("right_wheel_joint"));
  DRAKE_DEMAND(name_to_idx["right_wheel_joint"] == kWheelQStart);
  DRAKE_DEMAND(name_to_idx.count("lift_joint"));
  DRAKE_DEMAND(name_to_idx["lift_joint"] == kLiftQStart);
  DRAKE_DEMAND(name_to_idx.count("j2n6s300_joint_1"));
  DRAKE_DEMAND(name_to_idx["j2n6s300_joint_1"] == kArmQStart);
  DRAKE_DEMAND(name_to_idx.count("j2n6s300_joint_finger_1"));
  DRAKE_DEMAND(name_to_idx["j2n6s300_joint_finger_1"] == kHandQStart);

  // todo(rcory) Verify the velocity indices somehow
}

void PrintOutRemyTree(const RigidBodyTree<double>& tree) {

  std::map<std::string, int> name_to_idx = tree.computePositionNameToIndexMap();

  std::cout <<"=============================="<<std::endl;
  for(auto& p: name_to_idx)
    std::cout << p.first << ':' << p.second << ' '<<std::endl;
  std::cout <<"=============================="<<std::endl;

}

void CreateTreeFromFloatingModelAtPose(const std::string& model_file_name,
                                       RigidBodyTreed* tree,
                                       const Eigen::Isometry3d& pose) {
  auto weld_to_frame = std::allocate_shared<RigidBodyFrame<double>>(
      Eigen::aligned_allocator<RigidBodyFrame<double>>(), "world", nullptr,
      pose);

  drake::parsers::urdf::AddModelInstanceFromUrdfFile(
      model_file_name, drake::multibody::joints::kQuaternion, weld_to_frame,
      tree);
}

//int AddFloatingModelInstance(
//    const std::string& model_path, const Eigen::Vector3d& xyz,
//    const Eigen::Vector3d& rpy = Eigen::Vector3d::Zero(),
//    RigidBodyTreed* tree) {
//
//  auto weld_to_frame = allocate_shared<RigidBodyFrame<double>>(
//      Eigen::aligned_allocator<RigidBodyFrame<double>>(), "world",
//      nullptr, xyz, rpy);
//
//  drake::parsers::urdf::AddModelInstanceFromUrdfFile(
//      FindResourceOrThrow(model_path), drake::multibody::joints::kQuaternion,
//      weld_to_frame, tree);
//}

}  // namespace Remy
}  // namespace examples
}  // namespace drake
