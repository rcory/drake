#include "drake/multibody/rigid_body_tree.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
enum class RotateBox {
  HomeDepotPaper,
  AmazonRubber
};

std::unique_ptr<RigidBodyTreed> ConstructDualArmAndBox(RotateBox box_type = RotateBox::HomeDepotPaper);

void AddSphereToBody(RigidBodyTreed* tree, int link_idx,
                     const Eigen::Vector3d& pt, const std::string& name,
                     double radius);

void VisualizePosture(RigidBodyTreed* tree,
                      const Eigen::Ref<const Eigen::VectorXd>& q_kuka1,
                      const Eigen::Ref<const Eigen::VectorXd>& q_kuka2,
                      const Eigen::Ref<const Eigen::VectorXd>& q_box);

}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
