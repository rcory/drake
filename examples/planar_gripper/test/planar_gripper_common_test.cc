#include "drake/examples/planar_gripper/planar_gripper_common.h"

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace drake {
namespace examples {
namespace planar_gripper {

class PlanarGripperCommonTest : public ::testing::Test {
 protected:
  PlanarGripperCommonTest()
      : gripper_full_file_name_(FindResourceOrThrow(
            "drake/examples/planar_gripper/planar_gripper.sdf")),
        brick_full_file_name_(FindResourceOrThrow(
            "drake/examples/planar_gripper/planar_brick.sdf")),
        plant_(MultibodyPlant<double>(1e-3)) {
    gripper_index_ = multibody::Parser(&plant_).AddModelFromFile(
        gripper_full_file_name_, "planar_gripper");
    WeldGripperFrames<double>(&plant_, math::RigidTransformd::Identity());

    // Adds the brick to be manipulated.
    auto brick_index = multibody::Parser(&plant_).AddModelFromFile(
        brick_full_file_name_, "brick");
    const multibody::Frame<double>& brick_base_frame =
        plant_.GetFrameByName("brick_base_link", brick_index);
    plant_.WeldFrames(plant_.world_frame(), brick_base_frame);
    plant_.Finalize();
  }
  const std::string gripper_full_file_name_;
  const std::string brick_full_file_name_;
  MultibodyPlant<double> plant_;
  multibody::ModelInstanceIndex gripper_index_;
};

TEST_F(PlanarGripperCommonTest, MapUserOrderedStateToPlantState) {
  systems::DiagramBuilder<double> builder;
  auto user_order_vec = GetPreferredGripperJointOrdering();
  auto dut = builder.AddSystem<MapUserOrderedStateToPlantState>(
      plant_, user_order_vec, gripper_index_);

  auto outport = builder.ExportOutput(dut->get_output_port(0));
  builder.ExportInput(dut->get_input_port(0));
  auto diagram = builder.Build();
  auto diagram_context = diagram->CreateDefaultContext();

  VectorX<double> user_state = VectorX<double>::Zero(12);
  user_state.head(6) << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;

  diagram->get_input_port(0).FixValue(diagram_context.get(), user_state);
  systems::State<double>& diagram_state = diagram_context->get_mutable_state();
  diagram->CalcUnrestrictedUpdate(*diagram_context, &diagram_state);

  auto plant_state =
      diagram->get_output_port(outport).Eval<systems::BasicVector<double>>(
          *diagram_context).get_value();

  // Create the state selector matrix.
  std::vector<multibody::JointIndex> joint_indices;
  joint_indices.reserve(user_order_vec.size());
  for (auto & iter : user_order_vec) {
    joint_indices.push_back(plant_.GetJointByName(iter).index());
  }
  MatrixX<double> Sx_inv =
      plant_.MakeStateSelectorMatrix(joint_indices).inverse();
  drake::log()->info("Sx_inv: \n{}", Sx_inv);

  EXPECT_TRUE(CompareMatrices(Sx_inv * user_state, plant_state));
}

TEST_F(PlanarGripperCommonTest, MapUserOrderedActuationToPlantActuation) {
  systems::DiagramBuilder<double> builder;
  auto user_order_vec = GetPreferredGripperJointOrdering();
  auto dut = builder.AddSystem<MapUserOrderedActuationToPlantActuation>(
      plant_, user_order_vec, gripper_index_);

  auto outport = builder.ExportOutput(dut->get_output_port(0));
  builder.ExportInput(dut->get_input_port(0));
  auto diagram = builder.Build();
  auto diagram_context = diagram->CreateDefaultContext();

  VectorX<double> user_actuation = VectorX<double>::Zero(6);
  user_actuation << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;

  diagram->get_input_port(0).FixValue(diagram_context.get(), user_actuation);
  systems::State<double>& diagram_state = diagram_context->get_mutable_state();
  diagram->CalcUnrestrictedUpdate(*diagram_context, &diagram_state);

  auto plant_actuation =
      diagram->get_output_port(outport).Eval<systems::BasicVector<double>>(
          *diagram_context).get_value();

  // Create the actuation selector matrix.
  std::vector<multibody::JointIndex> joint_indices;
  joint_indices.reserve(user_order_vec.size());
  for (auto & iter : user_order_vec) {
    joint_indices.push_back(plant_.GetJointByName(iter).index());
  }
  MatrixX<double> Su_inv =
      plant_.MakeActuatorSelectorMatrix(joint_indices).inverse();
  drake::log()->info("Sx_inv: \n{}", Su_inv);

  EXPECT_TRUE(CompareMatrices(Su_inv * user_actuation, plant_actuation));
}

TEST_F(PlanarGripperCommonTest, MakeStateSelectorMatrix) {
  std::vector<std::string> joint_names =
      GetPreferredGripperJointOrdering();
  auto Sx_out = MakeStateSelectorMatrix(plant_, joint_names);

  std::vector<multibody::JointIndex> joint_indices(joint_names.size());
  for (size_t i = 0; i < joint_names.size(); i++) {
    joint_indices[i] =  plant_.GetJointByName(joint_names[i]).index();
  }
  auto Sx = plant_.MakeStateSelectorMatrix(joint_indices);
  EXPECT_TRUE(CompareMatrices(Sx, Sx_out));
}

TEST_F(PlanarGripperCommonTest, PreferredJointOrdering) {
  std::vector<std::string> user_finger_joint_order =
      GetPreferredFingerJointOrdering();
  EXPECT_EQ(user_finger_joint_order.size(), kNumJointsPerFinger);

  std::vector<std::string> user_gripper_joint_order =
      GetPreferredGripperJointOrdering();
  EXPECT_EQ(user_gripper_joint_order.size(),
            plant_.num_positions(gripper_index_));
  for (auto& iter : user_gripper_joint_order) {
    EXPECT_TRUE(plant_.HasJointNamed(iter));
  }
}

TEST_F(PlanarGripperCommonTest, ReorderKeyframesTest) {
  const int kNumKeyframes = 4;
  // Include keyframes for gripper joints as well as planar brick joints.
  const int kNumPlanarBrickJoints = 3;
  MatrixX<double> keyframes = MatrixX<double>::Zero(
      kNumGripperJoints + kNumPlanarBrickJoints, kNumKeyframes);
  VectorX<double> unit_row = VectorX<double>::Ones(kNumKeyframes);

  // Create an arbitrary keyframe matrix.
  for (int i = 0; i < keyframes.rows(); i++) {
    keyframes.row(i) = unit_row * i;
  }

  // Set an arbitrary row ordering for the above keyframes.
  std::map<std::string, int> joint_name_to_row_index_map;
  joint_name_to_row_index_map["finger1_BaseJoint"] = 3;
  joint_name_to_row_index_map["finger2_BaseJoint"] = 2;
  joint_name_to_row_index_map["finger3_BaseJoint"] = 4;
  joint_name_to_row_index_map["finger1_MidJoint"] = 0;
  joint_name_to_row_index_map["finger3_MidJoint"] = 5;
  joint_name_to_row_index_map["finger2_MidJoint"] = 1;
  joint_name_to_row_index_map["brick_translate_y_joint"] = 6;
  joint_name_to_row_index_map["brick_translate_z_joint"] = 8;
  joint_name_to_row_index_map["brick_revolute_x_joint"] = 7;

  // Reorder the keyframes and save the new results.
  std::map<std::string, int> joint_name_to_row_index_map_new =
      joint_name_to_row_index_map;
  auto keyframes_new = ReorderKeyframesForPlant(
      plant_, keyframes, &joint_name_to_row_index_map_new);

  DRAKE_DEMAND(keyframes.rows() == keyframes_new.rows() &&
               keyframes.cols() == keyframes_new.cols());
  DRAKE_DEMAND(joint_name_to_row_index_map.size() ==
               joint_name_to_row_index_map_new.size());

  // Get the velocity index ordering.
  std::map<std::string, int> joint_name_to_vel_index_ordering =
      joint_name_to_row_index_map;
  for (auto iter = joint_name_to_vel_index_ordering.begin();
       iter != joint_name_to_vel_index_ordering.end(); iter++) {
    iter->second = plant_.GetJointByName(iter->first).velocity_start();
  }

  // Make sure the keyframes were ordered correctly.
  for (auto iter = joint_name_to_vel_index_ordering.begin();
       iter != joint_name_to_vel_index_ordering.end(); iter++) {
    EXPECT_TRUE(CompareMatrices(
        keyframes_new.row(iter->second),
        keyframes.row(joint_name_to_row_index_map[iter->first])));
  }

  // Test throw when keyframe rows and joint name to row map size don't match.
  MatrixX<double> bad_rows_keyframes = /* adds one extra row */
      MatrixX<double>::Zero(kNumGripperJoints + 1, kNumKeyframes);
  EXPECT_THROW(ReorderKeyframesForPlant(plant_, bad_rows_keyframes,
                                        &joint_name_to_row_index_map),
               std::runtime_error);

  // Test throw when keyframe rows and joint name to row map size match, but
  // keyframe rows does not match number of planar-gripper joints.
  std::map<std::string, int> bad_joint_name_to_row_index_map =
      joint_name_to_row_index_map;
  bad_joint_name_to_row_index_map["finger1_ExtraJoint"] = 6;
  EXPECT_THROW(ReorderKeyframesForPlant(plant_, bad_rows_keyframes,
                                        &bad_joint_name_to_row_index_map),
               std::runtime_error);

  // Test throw when plant positions don't match number of expected planar
  // gripper joints.
  MultibodyPlant<double> bad_plant(0.0);
  multibody::Parser(&bad_plant)
      .AddModelFromFile(gripper_full_file_name_, "gripper");
  WeldGripperFrames(&bad_plant);
  const std::string extra_model_name =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_brick.sdf");
  multibody::Parser(&bad_plant).AddModelFromFile(extra_model_name, "brick");
  bad_plant.Finalize();
  EXPECT_THROW(
      ReorderKeyframesForPlant(bad_plant, keyframes,
                               &joint_name_to_row_index_map_new),
      std::runtime_error);
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
