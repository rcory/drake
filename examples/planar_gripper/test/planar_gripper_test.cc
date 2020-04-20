#include "drake/examples/planar_gripper/planar_gripper.h"

#include <gtest/gtest.h>

#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace drake {
namespace examples {
namespace planar_gripper {
GTEST_TEST(TestPlanarGripper, constructor) {
  PlanarGripper dut(0.1, false);
  dut.SetupPlanarBrick("horizontal");
  dut.Finalize();
  // Test geometry IDs.
  const auto& inspector = dut.get_scene_graph().model_inspector();
  for (const Finger finger :
       {Finger::kFinger1, Finger::kFinger2, Finger::kFinger3}) {
    EXPECT_EQ(dut.fingertip_sphere_geometry_id(finger),
              inspector.GetGeometryIdByName(
                  dut.get_multibody_plant().GetBodyFrameIdOrThrow(
                      dut.get_multibody_plant()
                          .GetBodyByName(to_string(finger) + "_tip_link")
                          .index()),
                  geometry::Role::kProximity,
                  "planar_gripper::tip_sphere_collision"));
  }
  EXPECT_EQ(
      dut.brick_geometry_id(),
      inspector.GetGeometryIdByName(
          dut.get_multibody_plant().GetBodyFrameIdOrThrow(
              dut.get_multibody_plant().GetBodyByName("brick_link").index()),
          geometry::Role::kProximity, "brick::box_collision"));
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
