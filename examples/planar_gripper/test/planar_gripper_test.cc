#include "drake/examples/planar_gripper/planar_gripper.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/sensors/image_to_lcm_image_array_t.h"

namespace drake {
namespace examples {
namespace planar_gripper {
GTEST_TEST(TestPlanarGripper, constructor) {
  PlanarGripper dut(0.1, ControlType::kTorque, false /* no floor */);
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

GTEST_TEST(TestPlanarGripper, InverseDynamicsGains) {
  PlanarGripper planar_gripper(0.1, ControlType::kPosition,
                               false /* no floor */);

  // Create correct size matrix. And make sure constructor allocates gains
  // properly.
  Vector<double, kNumGripperJoints> Kp_good, Ki_good, Kd_good;
  EXPECT_NO_THROW(planar_gripper.GetInverseDynamicsControlGains(
      &Kp_good, &Ki_good, &Kd_good));

  // Set the gains.
  Kp_good.setConstant(1.2);
  Ki_good.setConstant(2.3);
  Kd_good.setConstant(4.5);
  EXPECT_NO_THROW(
      planar_gripper.SetInverseDynamicsControlGains(Kp_good, Ki_good, Kd_good));

  // Check the gains.
  Vector<double, kNumGripperJoints> Kp_ret, Ki_ret, Kd_ret;
  EXPECT_NO_THROW(
      planar_gripper.GetInverseDynamicsControlGains(&Kp_ret, &Ki_ret, &Kd_ret));
  EXPECT_TRUE(CompareMatrices(Kp_ret, Kp_good));
  EXPECT_TRUE(CompareMatrices(Ki_ret, Ki_good));
  EXPECT_TRUE(CompareMatrices(Kd_ret, Kd_good));

  // Create an incorrect size matrix.
  Vector<double, kNumGripperJoints - 1> Kp_bad, Ki_bad, Kd_bad;
  EXPECT_THROW(
      planar_gripper.GetInverseDynamicsControlGains(&Kp_bad, &Ki_bad, &Kd_bad),
      std::logic_error);
}

GTEST_TEST(TestPlanarGripper, DefaultCameraSetup) {
  systems::DiagramBuilder<double> builder;
  auto planar_gripper = builder.AddSystem<PlanarGripper>(
      0.1, ControlType::kPosition, false /* no floor */);
  planar_gripper->SetupPlanarBrick("horizontal");
  planar_gripper->Finalize();
  auto diagram = builder.Build();
  auto diagram_context = diagram->CreateDefaultContext();
  systems::Context<double>& planar_gripper_context =
      diagram->GetMutableSubsystemContext(*planar_gripper,
                                          diagram_context.get());

  const std::string& camera_name = planar_gripper->get_default_camera_name();
  const auto image_array_t =
      planar_gripper->GetOutputPort(camera_name + "_images")
          .Eval<robotlocomotion::image_array_t>(planar_gripper_context);
  // Confirm that there are two images.
  EXPECT_EQ(image_array_t.images.size(), 2);
}

GTEST_TEST(TestPlanarGripper, PositionControlSetup) {
  systems::DiagramBuilder<double> builder;
  auto planar_gripper = builder.AddSystem<PlanarGripper>(
      0.1, ControlType::kPosition, false /* no floor */);
  planar_gripper->SetupPlanarBrick("horizontal");
  planar_gripper->Finalize();
  auto diagram = builder.Build();
  auto diagram_context = diagram->CreateDefaultContext();

  VectorX<double> x_des(2 * kNumGripperJoints);
  // Set an arbitrary desired state.
  for (int i = 0; i < x_des.size(); i++) {
    x_des(i) = i * 0.1;
  }
  systems::Context<double>& planar_gripper_context =
      diagram->GetMutableSubsystemContext(*planar_gripper,
                                          diagram_context.get());
  planar_gripper->GetInputPort("desired_gripper_state")
      .FixValue(&planar_gripper_context, x_des);

  // Force a diagram update.
  systems::State<double>& diagram_state = diagram_context->get_mutable_state();
  diagram->CalcUnrestrictedUpdate(*diagram_context, &diagram_state);

  // Make sure the inverse dynamics controller's desired state input matches.
  const auto& id_controller =
      planar_gripper->GetSubsystemByName("inverse_dynamics_controller");
  const systems::Context<double>& id_controller_context =
      planar_gripper->GetSubsystemContext(id_controller,
                                          planar_gripper_context);
  systems::InputPortIndex id_controller_desired_state_index(1);
  const VectorX<double> xdes_out =
      id_controller
          .EvalVectorInput(id_controller_context,
                           id_controller_desired_state_index)
          ->get_value();
  EXPECT_TRUE(CompareMatrices(x_des, xdes_out));
}

GTEST_TEST(TestPlanarGripper, TorqueControlSetup) {
  systems::DiagramBuilder<double> builder;
  auto planar_gripper = builder.AddSystem<PlanarGripper>(
      0.1, ControlType::kTorque, false /* no floor */);
  planar_gripper->SetupPlanarBrick("horizontal");
  planar_gripper->Finalize();
  auto diagram = builder.Build();
  auto diagram_context = diagram->CreateDefaultContext();

  VectorX<double> u_des(kNumGripperJoints);
  // Set arbitrary actuation values.
  for (int i = 0; i < u_des.size(); i++) {
    u_des(i) = i * 0.12;
  }
  systems::Context<double>& planar_gripper_context =
      diagram->GetMutableSubsystemContext(*planar_gripper,
                                          diagram_context.get());
  planar_gripper->GetInputPort("torque_control_u")
      .FixValue(&planar_gripper_context, u_des);

  // Force a diagram update.
  systems::State<double>& diagram_state = diagram_context->get_mutable_state();
  diagram->CalcUnrestrictedUpdate(*diagram_context, &diagram_state);

  // Make sure the plant's actuation input matches.
  const auto& plant = planar_gripper->GetSubsystemByName("multibody_plant");
  const systems::Context<double>& plant_context =
      planar_gripper->GetSubsystemContext(plant, planar_gripper_context);
  const VectorX<double> u_des_out =
      plant
          .EvalVectorInput(
              plant_context,
              plant.GetInputPort("planar_gripper_actuation").get_index())
          ->get_value();
  EXPECT_TRUE(CompareMatrices(u_des, u_des_out));
}

GTEST_TEST(TestPlanarGripper, HybridControlSetup) {
  systems::DiagramBuilder<double> builder;
  auto planar_gripper = builder.AddSystem<PlanarGripper>(
      0.1, ControlType::kHybrid, false /* no floor */);
  planar_gripper->SetupPlanarBrick("horizontal");
  planar_gripper->Finalize();
  auto diagram = builder.Build();
  auto diagram_context = diagram->CreateDefaultContext();
  systems::Context<double>& planar_gripper_context =
      diagram->GetMutableSubsystemContext(*planar_gripper,
                                          diagram_context.get());

  VectorX<double> x_des(2 * kNumGripperJoints);
  VectorX<double> u_des(kNumGripperJoints);

  // Set an arbitrary desired state.
  for (int i = 0; i < x_des.size(); i++) {
    x_des(i) = i * 0.1;
  }
  // Set arbitrary actuation values.
  for (int i = 0; i < u_des.size(); i++) {
    u_des(i) = i * 0.12;
  }
  for (int i = 0; i < kNumFingers; i++) {
    std::string finger_name = to_string_from_finger_num(i + 1);
    // Fix this finger's state input port.
    VectorX<double> finger_state(2 * kNumJointsPerFinger);
    finger_state.head(kNumJointsPerFinger) =
        x_des.segment(i * 2, kNumJointsPerFinger);
    finger_state.tail(kNumJointsPerFinger) =
        x_des.segment(kNumGripperJoints + (i * 2), kNumJointsPerFinger);
    planar_gripper
        ->GetInputPort(finger_name + "_desired_state")
        .FixValue(&planar_gripper_context, finger_state);

    // Fix this finger's u input port.
    VectorX<double> finger_u = u_des.segment(i * 2, kNumJointsPerFinger);
    planar_gripper
        ->GetInputPort(finger_name + "_torque_control_u")
        .FixValue(&planar_gripper_context, finger_u);

    // Fix this finger's control-type input port.
    ControlType control_type = ControlType::kPosition;
    planar_gripper
        ->GetInputPort(finger_name + "_control_type")
        .FixValue(&planar_gripper_context, control_type);
  }

  // Force a diagram update.
  systems::State<double>& diagram_state = diagram_context->get_mutable_state();
  diagram->CalcUnrestrictedUpdate(*diagram_context, &diagram_state);

  // Check the desired state appears at the input of the id_controller.
  // x_des is assumed to be in preferred joint ordering, and requires conversion
  // to joint velocity index ordering before comparing.
  auto pref_joint_ordering = GetPreferredGripperJointOrdering();
  EXPECT_TRUE(pref_joint_ordering.size() == kNumGripperJoints);
  std::map<std::string, double> joint_pos_value_map;
  std::map<std::string, double> joint_vel_value_map;
  for (auto iter = pref_joint_ordering.begin();
       iter != pref_joint_ordering.end(); iter++) {
    joint_pos_value_map[*iter] = x_des(iter - pref_joint_ordering.begin());
    joint_vel_value_map[*iter] =
        x_des(kNumGripperJoints + (iter - pref_joint_ordering.begin()));
  }
  VectorX<double> x_des_plant_ordering(2 * kNumGripperJoints);
  x_des_plant_ordering.head(kNumGripperJoints) =
      planar_gripper->MakeGripperPositionVector(joint_pos_value_map);
  x_des_plant_ordering.tail(kNumGripperJoints) =
      planar_gripper->MakeGripperVelocityVector(joint_vel_value_map);
  EXPECT_TRUE(x_des_plant_ordering.size() == (2 * kNumGripperJoints));
  const auto& id_controller =
      planar_gripper->GetSubsystemByName("inverse_dynamics_controller");
  const systems::Context<double>& id_controller_context =
      planar_gripper->GetSubsystemContext(id_controller,
                                          planar_gripper_context);
  systems::InputPortIndex id_controller_desired_state_index(1);
  const VectorX<double> xdes_id_controller =
      id_controller
          .EvalVectorInput(id_controller_context,
                           id_controller_desired_state_index)
          ->get_value();
  EXPECT_TRUE(CompareMatrices(x_des_plant_ordering, xdes_id_controller));

  // Check that the computed u of inverse dynamics control shows up at the
  // actuation input port of the MBP.
  const auto& force_to_actuation =
      planar_gripper->GetSubsystemByName("force_to_actuation_ordering");
  const systems::Context<double>& force_to_actuation_context =
      planar_gripper->GetSubsystemContext(force_to_actuation,
                                          planar_gripper_context);
  const VectorX<double> id_computed_u =
      force_to_actuation.GetOutputPort("u")
          .Eval<systems::BasicVector<double>>(force_to_actuation_context)
          .get_value();
  const auto& plant = planar_gripper->GetSubsystemByName("multibody_plant");
  const systems::Context<double>& plant_context =
      planar_gripper->GetSubsystemContext(plant, planar_gripper_context);
  VectorX<double> u_plant = plant.EvalVectorInput(
      plant_context,
      plant.GetInputPort("planar_gripper_actuation").get_index())->get_value();
  EXPECT_TRUE(CompareMatrices(id_computed_u, u_plant));

  // Now change the hybrid control switch to torque control.
  for (int i = 0; i < kNumFingers; i++) {
    std::string finger_name = to_string_from_finger_num(i + 1);
    // Fix this finger's control-type input port.
    ControlType control_type = ControlType::kTorque;
    planar_gripper
        ->GetInputPort(finger_name + "_control_type")
        .FixValue(&planar_gripper_context, control_type);
  }

  // Force a diagram update.
  diagram->CalcUnrestrictedUpdate(*diagram_context, &diagram_state);

  // Check that the torque control u input shows up at the actuation input port
  // of the MBP.
  // First, reorder the u vector into MBP joint actuator index ordering.
  std::vector<std::string> preferred_joint_ordering =
      GetPreferredGripperJointOrdering();
  std::vector<multibody::JointIndex> joint_index_vector;
  for (const auto& iter : preferred_joint_ordering) {
    joint_index_vector.push_back(
        planar_gripper->get_multibody_plant().GetJointByName(iter).index());
  }
  // Create the Sᵤ matrix.
  auto actuation_selector_matrix_ =
      planar_gripper->get_multibody_plant().MakeActuatorSelectorMatrix(
          joint_index_vector);
  u_plant = plant.EvalVectorInput(
      plant_context,
      plant.GetInputPort("planar_gripper_actuation").get_index())->get_value();
  EXPECT_TRUE(CompareMatrices(actuation_selector_matrix_ * u_des, u_plant));
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
