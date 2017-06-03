#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_rotate_box_planner.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
int DoMain() {
  auto tree = ConstructDualArmAndBox();

  return 0;
}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main() {
  return drake::examples::kuka_iiwa_arm::DoMain();
}