#include "drake/examples/planar_gripper/planar_gripper.h"

#include <gtest/gtest.h>

#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/solve.h"

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

GTEST_TEST(TestPlanarGripper, GetClosestFacesToFinger) {
  systems::DiagramBuilder<double> builder;
  auto planar_gripper = builder.AddSystem<PlanarGripper>(0.1, false);
  planar_gripper->SetupPlanarBrick("horizontal");
  planar_gripper->Finalize();
  auto diagram = builder.Build();
  auto diagram_context = diagram->CreateDefaultContext();
  auto plant_context = &diagram->GetMutableSubsystemContext(
      planar_gripper->get_multibody_plant(), diagram_context.get());

  // Solve an IK problem such that finger 1 is closest to face +Z, and finger 1
  // is outside of the brick.
  multibody::InverseKinematics ik1(planar_gripper->get_multibody_plant(),
                                   plant_context);

  const auto& inspector = planar_gripper->get_scene_graph().model_inspector();
  const auto finger1_sphere_geometry_id =
      planar_gripper->fingertip_sphere_geometry_id(Finger::kFinger1);
  // p_L2S is the position of the fingertip sphere center S in the link 2 frame
  // L2.
  const Eigen::Vector3d p_L2S =
      inspector.GetPoseInFrame(finger1_sphere_geometry_id).translation();
  const auto& brick_frame = planar_gripper->get_multibody_plant()
                                .GetBodyByName("brick_link")
                                .body_frame();
  ik1.AddPositionConstraint(planar_gripper->get_multibody_plant()
                                .GetBodyByName("finger1_tip_link")
                                .body_frame(),
                            p_L2S, brick_frame,
                            Eigen::Vector3d(0., -0.05, 0.08),
                            Eigen::Vector3d(0., 0.05, 0.2));
  Eigen::VectorXd x_init =
      0.1 * Eigen::VectorXd::Ones(
                planar_gripper->get_multibody_plant().num_positions());
  auto result = solvers::Solve(ik1.prog(), x_init);
  assert(result.is_success());
  EXPECT_EQ(
      planar_gripper->GetClosestFacesToFinger(*plant_context, Finger::kFinger1),
      std::unordered_set<BrickFace>({BrickFace::kPosZ}));

  // Solve an IK problem such that finger 1 sphere is inside the brick with -z
  // as the closest face.
  multibody::InverseKinematics ik2(planar_gripper->get_multibody_plant(),
                                   plant_context);
  ik2.AddPositionConstraint(planar_gripper->get_multibody_plant()
                                .GetBodyByName("finger1_tip_link")
                                .body_frame(),
                            p_L2S, brick_frame,
                            Eigen::Vector3d(0., -0.03, -0.05),
                            Eigen::Vector3d(0., 0.03, -0.04));
  result = solvers::Solve(ik2.prog(), x_init);
  assert(result.is_success());
  EXPECT_EQ(
      planar_gripper->GetClosestFacesToFinger(*plant_context, Finger::kFinger1),
      std::unordered_set<BrickFace>({BrickFace::kNegZ}));

  // Solve an IK problem such that finger 1 sphere is closest to the corner
  // neighbouring +z and -y faces.
  multibody::InverseKinematics ik3(planar_gripper->get_multibody_plant(),
                                   plant_context);
  ik3.AddPositionConstraint(planar_gripper->get_multibody_plant()
                                .GetBodyByName("finger1_tip_link")
                                .body_frame(),
                            p_L2S, brick_frame, Eigen::Vector3d(0., -0.2, 0.05),
                            Eigen::Vector3d(0., -0.05, 0.2));
  result = solvers::Solve(ik3.prog(), x_init);
  assert(result.is_success());
  EXPECT_EQ(
      planar_gripper->GetClosestFacesToFinger(*plant_context, Finger::kFinger1),
      std::unordered_set<BrickFace>({BrickFace::kPosZ, BrickFace::kNegY}));
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
