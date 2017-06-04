#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_rotate_box_planner.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
int DoMain() {

  auto construct_result = ConstructDualArmAndBox();
  auto tree = std::move(std::get<0>(construct_result));
  auto model_instance_id_table = std::get<1>(construct_result);
  for (const auto& model : model_instance_id_table) {
    std::cout << model.first << " id: " << model.second <<std::endl;
  }

  return 0;
}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main() {
  return drake::examples::kuka_iiwa_arm::DoMain();
}