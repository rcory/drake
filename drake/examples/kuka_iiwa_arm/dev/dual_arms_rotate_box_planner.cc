#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_rotate_box_planner.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
std::unique_ptr<RigidBodyTreed> ConstructDualArmAndBox() {
  std::unique_ptr<RigidBodyTree<double>> rigid_body_tree =
      std::make_unique<RigidBodyTree<double>>();
  const std::string model_path = drake::GetDrakePath() +
      "/manipulation/models/iiwa_description/urdf/"
          "iiwa14_polytope_collision.urdf";

  
}
}
}
}