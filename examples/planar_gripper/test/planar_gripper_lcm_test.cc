#include "drake/examples/planar_gripper/planar_gripper_lcm.h"

#include <gtest/gtest.h>

#include "drake/lcmt_planar_gripper_command.hpp"
#include "drake/lcmt_planar_gripper_status.hpp"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace examples {
namespace planar_gripper {
namespace {

constexpr int kNumFingers = 1;

// Test that encoding and decoding perform inverse operations by setting some
// values, encoding them, decoding them, and then verifying we get back what we
// put in.
GTEST_TEST(GripperLcmTest, GripperCommandPassthroughTest) {
  systems::DiagramBuilder<double> builder;
  auto command_encoder = builder.AddSystem<GripperCommandEncoder>(kNumFingers);
  auto command_decoder = builder.AddSystem<GripperCommandDecoder>(kNumFingers);
  builder.Connect(command_decoder->get_state_output_port(),
                  command_encoder->get_state_input_port());
  builder.Connect(command_decoder->get_torques_output_port(),
                  command_encoder->get_torques_input_port());
  builder.ExportInput(command_decoder->get_input_port(0));
  const systems::OutputPortIndex command_enc_output =
      builder.ExportOutput(command_encoder->get_output_port(0));
  auto diagram = builder.Build();

  std::unique_ptr<systems::Context<double>> context =
      diagram->CreateDefaultContext();

  lcmt_planar_gripper_command command{};
  command.num_fingers = kNumFingers;
  command.finger_command.resize(kNumFingers);
  auto& fcommand_in = command.finger_command[0];
  fcommand_in.joint_position[0] = 0.1;
  fcommand_in.joint_position[1] = 0.2;
  fcommand_in.joint_velocity[0] = 0.3;
  fcommand_in.joint_velocity[1] = 0.4;
  fcommand_in.joint_torque[0] = 0.5;
  fcommand_in.joint_torque[1] = 0.6;

  diagram->get_input_port(0).FixValue(context.get(), command);

  std::unique_ptr<systems::DiscreteValues<double>> update =
      diagram->AllocateDiscreteVariables();
  update->SetFrom(context->get_mutable_discrete_state());
  diagram->CalcDiscreteVariableUpdates(*context, update.get());
  context->get_mutable_discrete_state().SetFrom(*update);

  lcmt_planar_gripper_command command_out =
      diagram->get_output_port(command_enc_output)
          .Eval<lcmt_planar_gripper_command>(*context);
  auto& fcommand_out = command_out.finger_command[0];

  ASSERT_EQ(command.num_fingers, command_out.num_fingers);
  ASSERT_EQ(fcommand_in.joint_position[0], fcommand_out.joint_position[0]);
  ASSERT_EQ(fcommand_in.joint_position[1], fcommand_out.joint_position[1]);
  ASSERT_EQ(fcommand_in.joint_velocity[0], fcommand_out.joint_velocity[0]);
  ASSERT_EQ(fcommand_in.joint_velocity[1], fcommand_out.joint_velocity[1]);
  ASSERT_EQ(fcommand_in.joint_torque[0], fcommand_out.joint_torque[0]);
  ASSERT_EQ(fcommand_in.joint_torque[1], fcommand_out.joint_torque[1]);
}

// Test that encoding and decoding perform inverse operations by setting some
// values, encoding them, decoding them, and then verifying we get back what we
// put in.
GTEST_TEST(GripperLcmTest, GripperStatusPassthroughTest) {
  systems::DiagramBuilder<double> builder;
  auto status_encoder = builder.AddSystem<GripperStatusEncoder>(kNumFingers);
  auto status_decoder = builder.AddSystem<GripperStatusDecoder>(kNumFingers);
  builder.Connect(status_decoder->get_state_output_port(),
                  status_encoder->get_state_input_port());
  builder.Connect(status_decoder->get_force_output_port(),
                  status_encoder->get_force_input_port());
  builder.ExportInput(status_decoder->get_input_port(0));
  const systems::OutputPortIndex status_enc_output =
      builder.ExportOutput(status_encoder->get_output_port(0));
  auto diagram = builder.Build();

  std::unique_ptr<systems::Context<double>> context =
      diagram->CreateDefaultContext();

  lcmt_planar_gripper_status status{};
  status.num_fingers = kNumFingers;
  status.finger_status.resize(kNumFingers);
  auto& fstatus_in = status.finger_status[0];
  fstatus_in.joint_position[0] = 0.1;
  fstatus_in.joint_position[1] = 0.2;
  fstatus_in.joint_velocity[0] = 0.3;
  fstatus_in.joint_velocity[1] = 0.4;
  fstatus_in.fingertip_force.fx = 0;
  fstatus_in.fingertip_force.fy = 0.5;
  fstatus_in.fingertip_force.fz = 0.6;

  diagram->get_input_port(0).FixValue(context.get(), status);

  std::unique_ptr<systems::DiscreteValues<double>> update =
      diagram->AllocateDiscreteVariables();
  update->SetFrom(context->get_mutable_discrete_state());
  diagram->CalcDiscreteVariableUpdates(*context, update.get());
  context->get_mutable_discrete_state().SetFrom(*update);

  lcmt_planar_gripper_status status_out =
      diagram->get_output_port(status_enc_output)
          .Eval<lcmt_planar_gripper_status>(*context);
  auto& fstatus_out = status_out.finger_status[0];

  ASSERT_EQ(fstatus_in.joint_position[0], fstatus_out.joint_position[0]);
  ASSERT_EQ(fstatus_in.joint_position[1], fstatus_out.joint_position[1]);
  ASSERT_EQ(fstatus_in.joint_velocity[0], fstatus_out.joint_velocity[0]);
  ASSERT_EQ(fstatus_in.joint_velocity[1], fstatus_out.joint_velocity[1]);
  ASSERT_EQ(fstatus_in.fingertip_force.fx, fstatus_out.fingertip_force.fx);
  ASSERT_EQ(fstatus_in.fingertip_force.fy, fstatus_out.fingertip_force.fy);
  ASSERT_EQ(fstatus_in.fingertip_force.fz, fstatus_out.fingertip_force.fz);
}

GTEST_TEST(GripperLcmTest, QPBrickControlPassthroughTest) {
  systems::DiagramBuilder<double> builder;
  auto encoder = builder.AddSystem<QPBrickControlEncoder>();
  auto decoder =
      builder.AddSystem<QPBrickControlDecoder>(multibody::BodyIndex(3));
  builder.Connect(decoder->GetOutputPort("qp_brick_control"),
                  encoder->GetInputPort("qp_brick_control"));
  const systems::InputPortIndex decoder_input =
      builder.ExportInput(decoder->GetInputPort("spatial_forces_lcm"));
  const systems::OutputPortIndex encoder_output =
      builder.ExportOutput(encoder->GetOutputPort("spatial_forces_lcm"));
  auto diagram = builder.Build();

  std::unique_ptr<systems::Context<double>> context =
      diagram->CreateDefaultContext();

  lcmt_planar_manipuland_spatial_forces spatial_forces_lcm{};
  spatial_forces_lcm.manip_body_name = "brick_link";
  spatial_forces_lcm.num_forces = 2;

  auto& forces = spatial_forces_lcm.forces;
  forces.clear();
  forces.resize(2);
  ASSERT_EQ(forces.size(), spatial_forces_lcm.forces.size());
  forces[0].manip_body_name = "brick_link";
  forces[0].finger_name = "";
  forces[0].p_BoBq_B[0] = 0.1;
  forces[0].p_BoBq_B[1] = 0.2;
  forces[0].force_Bq_W[0] = 0.3;
  forces[0].force_Bq_W[1] = 0.4;
  forces[0].torque_Bq_W = .5;

  forces[1].manip_body_name = "brick_link";
  forces[1].finger_name = "";
  forces[1].p_BoBq_B[0] = 0.6;
  forces[1].p_BoBq_B[1] = 0.7;
  forces[1].force_Bq_W[0] = 0.8;
  forces[1].force_Bq_W[1] = 0.9;
  forces[1].torque_Bq_W = .1;

  diagram->get_input_port(decoder_input)
      .FixValue(context.get(), spatial_forces_lcm);

  auto& state = context->get_mutable_state();
  diagram->CalcUnrestrictedUpdate(*context, &state);

  lcmt_planar_manipuland_spatial_forces spatial_forces_lcm_out =
      diagram->get_output_port(encoder_output)
          .Eval<lcmt_planar_manipuland_spatial_forces>(*context);

  ASSERT_EQ(spatial_forces_lcm.utime, spatial_forces_lcm_out.utime);
  ASSERT_EQ(spatial_forces_lcm.manip_body_name,
            spatial_forces_lcm_out.manip_body_name);
  ASSERT_EQ(spatial_forces_lcm.num_forces, spatial_forces_lcm_out.num_forces);

  auto forces_out = spatial_forces_lcm_out.forces;
  for (int i = 0; i < 2; i++) {
    ASSERT_EQ(forces[i].manip_body_name, forces_out[i].manip_body_name);
    ASSERT_EQ(forces[i].finger_name, forces_out[i].finger_name);
    ASSERT_EQ(forces[i].p_BoBq_B[0], forces_out[i].p_BoBq_B[0]);
    ASSERT_EQ(forces[i].p_BoBq_B[1], forces_out[i].p_BoBq_B[1]);
    ASSERT_EQ(forces[i].force_Bq_W[0], forces_out[i].force_Bq_W[0]);
    ASSERT_EQ(forces[i].force_Bq_W[1], forces_out[i].force_Bq_W[1]);
    ASSERT_EQ(forces[i].torque_Bq_W, forces_out[i].torque_Bq_W);
  }
}

GTEST_TEST(GripperLcmTest, QPFingersControlPassthroughTest) {
  systems::DiagramBuilder<double> builder;
  const int kTestNumFingers = 3;
  auto encoder = builder.AddSystem<QPFingersControlEncoder>();
  auto decoder =
      builder.AddSystem<QPFingersControlDecoder>(multibody::BodyIndex(3));
  builder.Connect(decoder->GetOutputPort("qp_fingers_control"),
                  encoder->GetInputPort("qp_fingers_control"));
  const systems::InputPortIndex decoder_input =
      builder.ExportInput(decoder->GetInputPort("spatial_forces_lcm"));
  const systems::OutputPortIndex encoder_output =
      builder.ExportOutput(encoder->GetOutputPort("spatial_forces_lcm"));
  auto diagram = builder.Build();

  std::unique_ptr<systems::Context<double>> context =
      diagram->CreateDefaultContext();

  lcmt_planar_manipuland_spatial_forces spatial_forces_lcm{};
  spatial_forces_lcm.manip_body_name = "brick_link";
  spatial_forces_lcm.num_forces = kTestNumFingers;

  auto& forces = spatial_forces_lcm.forces;
  forces.clear();
  forces.resize(kTestNumFingers);
  ASSERT_EQ(forces.size(), spatial_forces_lcm.forces.size());
  forces[0].manip_body_name = "brick_link";
  forces[0].finger_name = "finger1";
  forces[0].p_BoBq_B[0] = 0.1;
  forces[0].p_BoBq_B[1] = 0.2;
  forces[0].force_Bq_W[0] = 0.3;
  forces[0].force_Bq_W[1] = 0.4;
  forces[0].torque_Bq_W = .5;

  forces[1].manip_body_name = "brick_link";
  forces[1].finger_name = "finger2";
  forces[1].p_BoBq_B[0] = 0.6;
  forces[1].p_BoBq_B[1] = 0.7;
  forces[1].force_Bq_W[0] = 0.8;
  forces[1].force_Bq_W[1] = 0.9;
  forces[1].torque_Bq_W = .1;

  forces[2].manip_body_name = "brick_link";
  forces[2].finger_name = "finger3";
  forces[2].p_BoBq_B[0] = 0.11;
  forces[2].p_BoBq_B[1] = 0.22;
  forces[2].force_Bq_W[0] = 0.33;
  forces[2].force_Bq_W[1] = 0.44;
  forces[2].torque_Bq_W = .55;

  diagram->get_input_port(decoder_input)
      .FixValue(context.get(), spatial_forces_lcm);

  auto& state = context->get_mutable_state();
  diagram->CalcUnrestrictedUpdate(*context, &state);

  lcmt_planar_manipuland_spatial_forces spatial_forces_lcm_out =
      diagram->get_output_port(encoder_output)
          .Eval<lcmt_planar_manipuland_spatial_forces>(*context);

  ASSERT_EQ(spatial_forces_lcm.utime, spatial_forces_lcm_out.utime);
  ASSERT_EQ(spatial_forces_lcm.manip_body_name,
            spatial_forces_lcm_out.manip_body_name);
  ASSERT_EQ(spatial_forces_lcm.num_forces, spatial_forces_lcm_out.num_forces);
  ASSERT_EQ(spatial_forces_lcm.forces.size(),
            spatial_forces_lcm_out.forces.size());

  auto forces_out = spatial_forces_lcm_out.forces;
  for (auto& force_out : forces_out) {
    // Look for the correct index, since the intermediate vector ordering is not
    // necessarily the same as the input.
    int findex;
    if (force_out.finger_name == "finger1") {
      findex = 0;
    } else if (force_out.finger_name == "finger2") {
      findex = 1;
    } else if (force_out.finger_name == "finger3") {
      findex = 2;
    } else {
      throw std::logic_error("Unknown finger name found");
    }
    ASSERT_EQ(forces[findex].manip_body_name, force_out.manip_body_name);
    ASSERT_EQ(forces[findex].finger_name, force_out.finger_name);
    ASSERT_EQ(forces[findex].p_BoBq_B[0], force_out.p_BoBq_B[0]);
    ASSERT_EQ(forces[findex].p_BoBq_B[1], force_out.p_BoBq_B[1]);
    ASSERT_EQ(forces[findex].force_Bq_W[0], force_out.force_Bq_W[0]);
    ASSERT_EQ(forces[findex].force_Bq_W[1], force_out.force_Bq_W[1]);
    ASSERT_EQ(forces[findex].torque_Bq_W, force_out.torque_Bq_W);
  }
}

GTEST_TEST(GripperLcmTest, QPEstimatedStatePassthroughTest) {
  systems::DiagramBuilder<double> builder;
  const int kTestNumStates = 2;
  auto encoder = builder.AddSystem<QPEstimatedStateEncoder>(kTestNumStates);
  auto decoder = builder.AddSystem<QPEstimatedStateDecoder>(kTestNumStates);
  builder.Connect(decoder->GetOutputPort("qp_estimated_plant_state"),
                  encoder->GetInputPort("qp_estimated_plant_state"));
  const systems::InputPortIndex decoder_input =
      builder.ExportInput(decoder->GetInputPort("planar_plant_state_lcm"));
  const systems::OutputPortIndex encoder_output =
      builder.ExportOutput(encoder->GetOutputPort("planar_plant_state_lcm"));
  auto diagram = builder.Build();

  std::unique_ptr<systems::Context<double>> context =
      diagram->CreateDefaultContext();

  lcmt_planar_plant_state planar_plant_state_lcm{};
  planar_plant_state_lcm.num_states = kTestNumStates;
  planar_plant_state_lcm.plant_state.resize(kTestNumStates);
  DRAKE_DEMAND(planar_plant_state_lcm.plant_state.size() == kTestNumStates);
  planar_plant_state_lcm.plant_state[0] = 1.0;
  planar_plant_state_lcm.plant_state[1] = 2.0;

  diagram->get_input_port(decoder_input)
      .FixValue(context.get(), planar_plant_state_lcm);

  //  auto& state = context->get_mutable_state();
  //  diagram->CalcUnrestrictedUpdate(*context, &state);
  std::unique_ptr<systems::DiscreteValues<double>> update =
      diagram->AllocateDiscreteVariables();
  update->SetFrom(context->get_mutable_discrete_state());
  diagram->CalcDiscreteVariableUpdates(*context, update.get());
  context->get_mutable_discrete_state().SetFrom(*update);

  lcmt_planar_plant_state planar_plant_state_lcm_out =
      diagram->get_output_port(encoder_output)
          .Eval<lcmt_planar_plant_state>(*context);

  ASSERT_EQ(planar_plant_state_lcm.num_states,
            planar_plant_state_lcm_out.num_states);
  ASSERT_EQ(kTestNumStates, planar_plant_state_lcm_out.plant_state.size());
  ASSERT_EQ(planar_plant_state_lcm.plant_state[0],
            planar_plant_state_lcm_out.plant_state[0]);
  ASSERT_EQ(planar_plant_state_lcm.plant_state[1],
            planar_plant_state_lcm_out.plant_state[1]);
}

GTEST_TEST(GripperLcmTest, QPFingerFaceAssignmentsPassthroughTest) {
  systems::DiagramBuilder<double> builder;
  const int kTestNumFingers = 2;
  auto encoder = builder.AddSystem<QPFingerFaceAssignmentsEncoder>();
  auto decoder = builder.AddSystem<QPFingerFaceAssignmentsDecoder>();
  builder.Connect(decoder->GetOutputPort("qp_finger_face_assignments"),
                  encoder->GetInputPort("qp_finger_face_assignments"));
  const systems::InputPortIndex decoder_input =
      builder.ExportInput(decoder->GetInputPort("finger_face_assignments_lcm"));
  const systems::OutputPortIndex encoder_output = builder.ExportOutput(
      encoder->GetOutputPort("finger_face_assignments_lcm"));
  auto diagram = builder.Build();

  std::unique_ptr<systems::Context<double>> context =
      diagram->CreateDefaultContext();

  lcmt_planar_gripper_finger_face_assignments finger_face_assignments_lcm{};
  finger_face_assignments_lcm.num_fingers = kTestNumFingers;
  finger_face_assignments_lcm.finger_face_assignments.resize(kTestNumFingers);
  DRAKE_DEMAND(finger_face_assignments_lcm.finger_face_assignments.size() ==
               kTestNumFingers);

  auto& assignments = finger_face_assignments_lcm.finger_face_assignments;
  assignments[0].finger_name = "finger1";
  assignments[0].brick_face_name = "NegY";
  assignments[0].p_BoBq_B[0] = .1;
  assignments[0].p_BoBq_B[1] = .2;

  assignments[1].finger_name = "finger2";
  assignments[1].brick_face_name = "PosY";
  assignments[1].p_BoBq_B[0] = .3;
  assignments[1].p_BoBq_B[1] = .4;

  diagram->get_input_port(decoder_input)
      .FixValue(context.get(), finger_face_assignments_lcm);

  auto& state = context->get_mutable_state();
  diagram->CalcUnrestrictedUpdate(*context, &state);

  auto finger_face_assignments_lcm_out =
      diagram->get_output_port(encoder_output)
          .Eval<lcmt_planar_gripper_finger_face_assignments>(*context);

  ASSERT_EQ(finger_face_assignments_lcm.num_fingers,
            finger_face_assignments_lcm_out.num_fingers);
  ASSERT_EQ(finger_face_assignments_lcm.finger_face_assignments.size(),
            finger_face_assignments_lcm_out.finger_face_assignments.size());

  auto& assignment0 = finger_face_assignments_lcm.finger_face_assignments[0];
  auto& assignment0_out =
      finger_face_assignments_lcm_out.finger_face_assignments[0];
  auto& assignment1 = finger_face_assignments_lcm.finger_face_assignments[1];
  auto& assignment1_out =
      finger_face_assignments_lcm_out.finger_face_assignments[1];

  // Because the intermediate form is an unordered_map, we can't guarantee
  // the ordering during encoding, so we allow for either outcome.
  if (assignment0_out.finger_name == "finger1") {
    DRAKE_DEMAND(assignment1_out.finger_name == "finger2");
    ASSERT_EQ(assignment0.finger_name, assignment0_out.finger_name);
    ASSERT_EQ(assignment0.brick_face_name, assignment0_out.brick_face_name);
    ASSERT_EQ(assignment0.p_BoBq_B[0], assignment0_out.p_BoBq_B[0]);
    ASSERT_EQ(assignment0.p_BoBq_B[1], assignment0_out.p_BoBq_B[1]);

    ASSERT_EQ(assignment1.finger_name, assignment1_out.finger_name);
    ASSERT_EQ(assignment1.brick_face_name, assignment1_out.brick_face_name);
    ASSERT_EQ(assignment1.p_BoBq_B[0], assignment1_out.p_BoBq_B[0]);
    ASSERT_EQ(assignment1.p_BoBq_B[1], assignment1_out.p_BoBq_B[1]);
  } else {
    DRAKE_DEMAND(assignment0_out.finger_name == "finger2");
    DRAKE_DEMAND(assignment1_out.finger_name == "finger1");
    ASSERT_EQ(assignment0.finger_name, assignment1_out.finger_name);
    ASSERT_EQ(assignment0.brick_face_name, assignment1_out.brick_face_name);
    ASSERT_EQ(assignment0.p_BoBq_B[0], assignment1_out.p_BoBq_B[0]);
    ASSERT_EQ(assignment0.p_BoBq_B[1], assignment1_out.p_BoBq_B[1]);

    ASSERT_EQ(assignment1.finger_name, assignment0_out.finger_name);
    ASSERT_EQ(assignment1.brick_face_name, assignment0_out.brick_face_name);
    ASSERT_EQ(assignment1.p_BoBq_B[0], assignment0_out.p_BoBq_B[0]);
    ASSERT_EQ(assignment1.p_BoBq_B[1], assignment0_out.p_BoBq_B[1]);
  }
}

GTEST_TEST(GripperLcmTest, QPBrickDesiredPassthroughTest) {
  systems::DiagramBuilder<double> builder;
  const int kTestNumBrickStates = 6;
  const int kTestNumBrickAccels = 3;
  auto encoder = builder.AddSystem<QPBrickDesiredEncoder>(kTestNumBrickStates,
                                                          kTestNumBrickAccels);
  auto decoder = builder.AddSystem<QPBrickDesiredDecoder>(kTestNumBrickStates,
                                                          kTestNumBrickAccels);
  builder.Connect(decoder->GetOutputPort("qp_desired_brick_state"),
                  encoder->GetInputPort("qp_desired_brick_state"));
  builder.Connect(decoder->GetOutputPort("qp_desired_brick_accel"),
                  encoder->GetInputPort("qp_desired_brick_accel"));
  const systems::InputPortIndex decoder_input =
      builder.ExportInput(decoder->GetInputPort("brick_desired_lcm"));
  const systems::OutputPortIndex encoder_output =
      builder.ExportOutput(encoder->GetOutputPort("brick_desired_lcm"));
  auto diagram = builder.Build();

  std::unique_ptr<systems::Context<double>> context =
      diagram->CreateDefaultContext();

  lcmt_planar_manipuland_desired brick_desired_lcm{};
  brick_desired_lcm.num_states = kTestNumBrickStates;
  brick_desired_lcm.num_accels = kTestNumBrickAccels;
  auto& desired_state = brick_desired_lcm.desired_state;
  auto& desired_accel = brick_desired_lcm.desired_accel;
  desired_state.resize(kTestNumBrickStates);
  desired_accel.resize(kTestNumBrickAccels);
  desired_state[0] = 0.1;
  desired_state[1] = 0.2;
  desired_state[2] = 0.3;
  desired_state[3] = 0.4;
  desired_state[4] = 0.5;
  desired_state[5] = 0.6;
  desired_accel[0] = 0.11;
  desired_accel[1] = 0.22;
  desired_accel[2] = 0.33;

  diagram->get_input_port(decoder_input)
      .FixValue(context.get(), brick_desired_lcm);

  std::unique_ptr<systems::DiscreteValues<double>> update =
      diagram->AllocateDiscreteVariables();
  update->SetFrom(context->get_mutable_discrete_state());
  diagram->CalcDiscreteVariableUpdates(*context, update.get());
  context->get_mutable_discrete_state().SetFrom(*update);

  auto brick_desired_lcm_out =
      diagram->get_output_port(encoder_output)
          .Eval<lcmt_planar_manipuland_desired>(*context);
  ASSERT_EQ(brick_desired_lcm.num_states, brick_desired_lcm_out.num_states);
  ASSERT_EQ(brick_desired_lcm.num_accels, brick_desired_lcm_out.num_accels);
  ASSERT_EQ(brick_desired_lcm.desired_state.size(),
            brick_desired_lcm_out.desired_state.size());
  ASSERT_EQ(brick_desired_lcm.desired_accel.size(),
            brick_desired_lcm_out.desired_accel.size());
  for (int i = 0; i < brick_desired_lcm.num_states; i++) {
    ASSERT_EQ(brick_desired_lcm.desired_state[i],
              brick_desired_lcm_out.desired_state[i]);
  }
  for (int i = 0; i < brick_desired_lcm.num_accels; i++) {
    ASSERT_EQ(brick_desired_lcm.desired_accel[i],
              brick_desired_lcm_out.desired_accel[i]);
  }
}

}  // namespace
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
