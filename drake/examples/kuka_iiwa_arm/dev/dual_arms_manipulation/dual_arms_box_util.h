#include "drake/multibody/rigid_body_tree.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
std::unique_ptr<RigidBodyTreed> ConstructDualArmAndBox();

void VisualizePosture(RigidBodyTreed* tree,
                      const Eigen::Ref<const Eigen::VectorXd>& q_kuka1,
                      const Eigen::Ref<const Eigen::VectorXd>& q_kuka2,
                      const Eigen::Ref<const Eigen::VectorXd>& q_box);

}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
