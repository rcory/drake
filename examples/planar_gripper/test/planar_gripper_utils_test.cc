#include "drake/examples/planar_gripper/planar_gripper_utils.h"

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/examples/planar_gripper/planar_gripper.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace examples {
namespace planar_gripper {
GTEST_TEST(TestPlanarGripper, GetClosestFacesToFinger) {
  // Test GetClosestFacesToFinger and also FingerFaceAssigner class.
  systems::DiagramBuilder<double> builder;
  auto planar_gripper = builder.AddSystem<PlanarGripper>(
      0.1, ControlType::kTorque, false /* no floor */);
  planar_gripper->SetupPlanarBrick("horizontal");
  planar_gripper->Finalize();
  auto finger_face_assigner = builder.AddSystem<FingerFaceAssigner>(
      planar_gripper->get_multibody_plant(), planar_gripper->get_scene_graph());
  builder.Connect(planar_gripper->GetOutputPort("scene_graph_query"),
                  finger_face_assigner->get_geometry_query_input_port());
  builder.Connect(planar_gripper->GetOutputPort("contact_results"),
                  finger_face_assigner->GetInputPort("contact_results"));
  builder.Connect(planar_gripper->GetOutputPort("plant_state"),
                  finger_face_assigner->GetInputPort("plant_state"));
  auto diagram = builder.Build();
  auto diagram_context = diagram->CreateDefaultContext();
  auto plant_context = &diagram->GetMutableSubsystemContext(
      planar_gripper->get_multibody_plant(), diagram_context.get());

  plant_context->FixInputPort(planar_gripper->get_multibody_plant()
                                  .get_actuation_input_port()
                                  .get_index(),
                              Eigen::VectorXd::Zero(kNumGripperJoints));

  // Solve an IK problem such that finger 1 is closest to face +Z, and finger 1
  // is outside of the brick.
  multibody::InverseKinematics ik1(planar_gripper->get_multibody_plant(),
                                   plant_context);

  const auto& inspector = planar_gripper->get_scene_graph().model_inspector();
  const auto finger1_sphere_geometry_id =
      planar_gripper->fingertip_sphere_geometry_id(Finger::kFinger1);
  // p_TS is the position of the fingertip sphere center S in the tip link frame
  // T.
  const Eigen::Vector3d p_TS =
      inspector.GetPoseInFrame(finger1_sphere_geometry_id).translation();
  const auto& brick_frame = planar_gripper->get_multibody_plant()
                                .GetBodyByName("brick_link")
                                .body_frame();
  const auto& finger1_tip_frame = planar_gripper->get_multibody_plant()
                                      .GetBodyByName("finger1_tip_link")
                                      .body_frame();
  ik1.AddPositionConstraint(finger1_tip_frame, p_TS, brick_frame,
                            Eigen::Vector3d(0., -0.05, 0.08),
                            Eigen::Vector3d(0., 0.05, 0.2));
  Eigen::VectorXd x_init =
      0.1 * Eigen::VectorXd::Ones(
                planar_gripper->get_multibody_plant().num_positions());
  auto result = solvers::Solve(ik1.prog(), x_init);
  assert(result.is_success());
  std::unordered_set<BrickFace> closest_faces;
  Eigen::Vector3d p_BCb;
  std::tie(closest_faces, p_BCb) = GetClosestFacesToFinger(
      planar_gripper->get_multibody_plant(), planar_gripper->get_scene_graph(),
      *plant_context, Finger::kFinger1);
  EXPECT_EQ(closest_faces, std::unordered_set<BrickFace>({BrickFace::kPosZ}));
  // position of finger tip sphere center S in the brick frame B.
  Eigen::Vector3d p_BS;
  planar_gripper->get_multibody_plant().CalcPointsPositions(
      *plant_context, finger1_tip_frame, p_TS, brick_frame, &p_BS);
  EXPECT_TRUE(CompareMatrices((p_BS - p_BCb).normalized(),
                              Eigen::Vector3d::UnitZ(),
                              std::numeric_limits<double>::epsilon()));
  // Also check if FingerFaceAssigner generates the right output.
  auto finger_face_assigner_output =
      finger_face_assigner->get_finger_face_assignments_output_port()
          .Eval<std::unordered_map<Finger, BrickFaceInfo>>(
              diagram->GetSubsystemContext(*finger_face_assigner,
                                           *diagram_context));
  EXPECT_EQ(finger_face_assigner_output.size(), kNumFingers);

  EXPECT_EQ(finger_face_assigner_output.at(Finger::kFinger1).brick_face,
            BrickFace::kPosZ);
  EXPECT_EQ(finger_face_assigner_output.at(Finger::kFinger1).p_BCb,
            Eigen::Vector2d(p_BCb.tail<2>()));
  EXPECT_EQ(finger_face_assigner_output.at(Finger::kFinger1).is_in_contact,
            false);

  // Solve an IK problem such that finger 1 sphere is inside the brick with -z
  // as the closest face.
  multibody::InverseKinematics ik2(planar_gripper->get_multibody_plant(),
                                   plant_context);
  ik2.AddPositionConstraint(finger1_tip_frame, p_TS, brick_frame,
                            Eigen::Vector3d(0., -0.03, -0.05),
                            Eigen::Vector3d(0., 0.03, -0.04));
  result = solvers::Solve(ik2.prog(), x_init);
  assert(result.is_success());
  std::tie(closest_faces, p_BCb) = GetClosestFacesToFinger(
      planar_gripper->get_multibody_plant(), planar_gripper->get_scene_graph(),
      *plant_context, Finger::kFinger1);
  EXPECT_EQ(closest_faces, std::unordered_set<BrickFace>({BrickFace::kNegZ}));
  planar_gripper->get_multibody_plant().CalcPointsPositions(
      *plant_context, finger1_tip_frame, p_TS, brick_frame, &p_BS);
  EXPECT_TRUE(CompareMatrices((p_BS - p_BCb).normalized(),
                              Eigen::Vector3d::UnitZ(), 1e-14));

  // Solve an IK problem such that finger 1 sphere is closest to the corner
  // neighbouring +z and -y faces.
  multibody::InverseKinematics ik3(planar_gripper->get_multibody_plant(),
                                   plant_context);
  ik3.AddPositionConstraint(finger1_tip_frame, p_TS, brick_frame,
                            Eigen::Vector3d(0., -0.2, 0.05),
                            Eigen::Vector3d(0., -0.05, 0.2));
  result = solvers::Solve(ik3.prog(), x_init);
  assert(result.is_success());
  std::tie(closest_faces, p_BCb) = GetClosestFacesToFinger(
      planar_gripper->get_multibody_plant(), planar_gripper->get_scene_graph(),
      *plant_context, Finger::kFinger1);
  EXPECT_EQ(closest_faces, std::unordered_set<BrickFace>(
                               {BrickFace::kPosZ, BrickFace::kNegY}));
  EXPECT_TRUE(CompareMatrices(p_BCb, Eigen::Vector3d(0, -0.05, 0.05)));
  finger_face_assigner_output =
      finger_face_assigner->get_finger_face_assignments_output_port()
          .Eval<std::unordered_map<Finger, BrickFaceInfo>>(
              diagram->GetSubsystemContext(*finger_face_assigner,
                                           *diagram_context));
  EXPECT_EQ(finger_face_assigner_output.size(), kNumFingers);
  EXPECT_TRUE(finger_face_assigner_output.at(Finger::kFinger1).brick_face ==
                  BrickFace::kPosZ ||
              finger_face_assigner_output.at(Finger::kFinger1).brick_face ==
                  BrickFace::kNegY);
  EXPECT_TRUE(CompareMatrices(
      finger_face_assigner_output.at(Finger::kFinger1).p_BCb, p_BCb.tail<2>()));
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
