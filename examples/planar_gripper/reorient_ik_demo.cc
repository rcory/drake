#include <memory>

#include "drake/examples/planar_gripper/gripper_brick.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"

namespace drake {
namespace examples {
int DoMain() {
  GripperBrickSystem<double> gripper_brick_system;
  return 0;
}
}  // namespace examples
}  // namespace drake

int main() { drake::examples::DoMain(); }
