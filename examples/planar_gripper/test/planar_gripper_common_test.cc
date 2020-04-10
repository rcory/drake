#include "drake/examples/planar_gripper/planar_gripper_common.h"

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace examples {
namespace planar_gripper {

GTEST_TEST(ReorderKeyframesTest, Test) {
  const int kNumKeyframes = 4;
  MatrixX<double> keyframes =
      MatrixX<double>::Zero(kNumGripperJoints, kNumKeyframes);
  VectorX<double> unit_row = VectorX<double>::Ones(kNumKeyframes);

  // Create an arbitrary keyframe matrix.
  for (int i = 0; i < keyframes.rows(); i++) {
    keyframes.row(i) = unit_row * i;
  }

  // Set an arbitrary row ordering for the above keyframes.
  std::map<std::string, int> finger_joint_name_to_row_index_map;
  finger_joint_name_to_row_index_map["finger1_BaseJoint"] = 3;
  finger_joint_name_to_row_index_map["finger2_BaseJoint"] = 2;
  finger_joint_name_to_row_index_map["finger3_BaseJoint"] = 4;
  finger_joint_name_to_row_index_map["finger1_MidJoint"] = 0;
  finger_joint_name_to_row_index_map["finger3_MidJoint"] = 5;
  finger_joint_name_to_row_index_map["finger2_MidJoint"] = 1;

  // Create the plant which defines the ordering.
  const std::string full_name =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_gripper.sdf");
  MultibodyPlant<double> plant(0.0);
  multibody::Parser(&plant).AddModelFromFile(full_name, "gripper");
  WeldGripperFrames(&plant);
  plant.Finalize();

  // Reorder the keyframes and save the new results.
  std::map<std::string, int> finger_joint_name_to_row_index_map_new =
      finger_joint_name_to_row_index_map;
  auto keyframes_new = ReorderKeyframesForPlant(
      plant, keyframes, &finger_joint_name_to_row_index_map_new);

  DRAKE_DEMAND(keyframes.rows() == keyframes_new.rows() &&
               keyframes.cols() == keyframes_new.cols());
  DRAKE_DEMAND(finger_joint_name_to_row_index_map.size() ==
               finger_joint_name_to_row_index_map_new.size());

  // Get the velocity index ordering.
  std::map<std::string, int> finger_joint_name_to_vel_index_ordering =
      finger_joint_name_to_row_index_map;
  for (auto iter = finger_joint_name_to_vel_index_ordering.begin();
       iter != finger_joint_name_to_vel_index_ordering.end(); iter++) {
    iter->second = plant.GetJointByName(iter->first).velocity_start();
  }

  // Make sure the keyframes were ordered correctly.
  for (auto iter = finger_joint_name_to_vel_index_ordering.begin();
       iter != finger_joint_name_to_vel_index_ordering.end(); iter++) {
    EXPECT_TRUE(CompareMatrices(
        keyframes_new.row(iter->second),
        keyframes.row(finger_joint_name_to_row_index_map[iter->first])));
  }

  // Test throw when keyframe rows and joint name to row map size don't match.
  MatrixX<double> bad_rows_keyframes = /* adds one extra row */
      MatrixX<double>::Zero(kNumGripperJoints + 1, kNumKeyframes);
  EXPECT_THROW(ReorderKeyframesForPlant(plant, bad_rows_keyframes,
                                        &finger_joint_name_to_row_index_map),
               std::runtime_error);

  // Test throw when keyframe rows and joint name to row map size match, but
  // keyframe rows does not match number of planar-gripper joints.
  std::map<std::string, int> bad_finger_joint_name_to_row_index_map =
      finger_joint_name_to_row_index_map;
  bad_finger_joint_name_to_row_index_map["finger1_ExtraJoint"] = 6;
  EXPECT_THROW(
      ReorderKeyframesForPlant(plant, bad_rows_keyframes,
                               &bad_finger_joint_name_to_row_index_map),
      std::runtime_error);

  // Test throw when plant positions don't match number of expected planar
  // gripper joints.
  MultibodyPlant<double> bad_plant(0.0);
  multibody::Parser(&bad_plant).AddModelFromFile(full_name, "gripper");
  WeldGripperFrames(&bad_plant);
  const std::string extra_model_name =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_brick.sdf");
  multibody::Parser(&bad_plant).AddModelFromFile(extra_model_name, "brick");
  bad_plant.Finalize();
  EXPECT_THROW(
      ReorderKeyframesForPlant(bad_plant, keyframes,
                               &finger_joint_name_to_row_index_map_new),
      std::runtime_error);
}

GTEST_TEST(TestGetClosestFacesToFinger, Test) {
  systems::DiagramBuilder<double> builder{};
  multibody::MultibodyPlant<double>* plant;
  geometry::SceneGraph<double>* scene_graph;
  std::tie(plant, scene_graph) =
      multibody::AddMultibodyPlantSceneGraph(&builder, 0.1);
  const std::string full_name =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_gripper.sdf");
  multibody::Parser(plant).AddModelFromFile(full_name, "planar_gripper");
  WeldGripperFrames(plant);
  multibody::Parser(plant).AddModelFromFile(
      FindResourceOrThrow("drake/examples/planar_gripper/planar_brick.sdf"),
      "brick");
  plant->Finalize();

  auto diagram = builder.Build();
  auto diagram_context = diagram->CreateDefaultContext();
  auto plant_context =
      &diagram->GetMutableSubsystemContext(*plant, diagram_context.get());

  // Solve an IK problem such that finger 1 is closest to face +Z, and finger 1
  // is outside of the brick.
  multibody::InverseKinematics ik1(*plant, plant_context);

  const auto& inspector = scene_graph->model_inspector();
  const auto finger1_sphere_geometry_id = inspector.GetGeometryIdByName(
      plant->GetBodyFrameIdOrThrow(
          plant->GetBodyByName("finger1_tip_link").index()),
      geometry::Role::kProximity, "planar_gripper::tip_sphere_collision");
  // p_L2S is the position of the fingertip sphere center S in the link 2 frame
  // L2.
  const Eigen::Vector3d p_L2S =
      inspector.GetPoseInFrame(finger1_sphere_geometry_id).translation();
  const auto& brick_frame = plant->GetBodyByName("brick_link").body_frame();
  ik1.AddPositionConstraint(
      plant->GetBodyByName("finger1_tip_link").body_frame(), p_L2S, brick_frame,
      Eigen::Vector3d(0., -0.05, 0.08), Eigen::Vector3d(0., 0.05, 0.2));
  Eigen::VectorXd x_init = 0.1 * Eigen::VectorXd::Ones(plant->num_positions());
  auto result = solvers::Solve(ik1.prog(), x_init);
  assert(result.is_success());
  EXPECT_EQ(GetClosestFacesToFinger(*plant, *scene_graph, *plant_context,
                                    Finger::kFinger1),
            std::unordered_set<BrickFace>({BrickFace::kPosZ}));

  // Solve an IK problem such that finger 1 sphere is inside the brick with -z
  // as the closest face.
  multibody::InverseKinematics ik2(*plant, plant_context);
  ik2.AddPositionConstraint(
      plant->GetBodyByName("finger1_tip_link").body_frame(), p_L2S, brick_frame,
      Eigen::Vector3d(0., -0.03, -0.05), Eigen::Vector3d(0., 0.03, -0.04));
  result = solvers::Solve(ik2.prog(), x_init);
  assert(result.is_success());
  EXPECT_EQ(GetClosestFacesToFinger(*plant, *scene_graph, *plant_context,
                                    Finger::kFinger1),
            std::unordered_set<BrickFace>({BrickFace::kNegZ}));

  // Solve an IK problem such that finger 1 sphere is closest to the corner
  // neighbouring +z and -y faces.
  multibody::InverseKinematics ik3(*plant, plant_context);
  ik3.AddPositionConstraint(
      plant->GetBodyByName("finger1_tip_link").body_frame(), p_L2S, brick_frame,
      Eigen::Vector3d(0., -0.2, 0.05), Eigen::Vector3d(0., -0.05, 0.2));
  result = solvers::Solve(ik3.prog(), x_init);
  assert(result.is_success());
  EXPECT_EQ(
      GetClosestFacesToFinger(*plant, *scene_graph, *plant_context,
                              Finger::kFinger1),
      std::unordered_set<BrickFace>({BrickFace::kPosZ, BrickFace::kNegY}));
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
