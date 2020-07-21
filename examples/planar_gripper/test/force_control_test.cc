#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/examples/planar_gripper/finger_brick.h"
#include "drake/examples/planar_gripper/finger_brick_control.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/plant/spatial_forces_to_lcm.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_value_source.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/zero_order_hold.h"

namespace drake {
namespace examples {
namespace planar_gripper {

using geometry::SceneGraph;
using multibody::AddMultibodyPlantSceneGraph;
using multibody::ContactResults;
using multibody::MultibodyPlant;
using multibody::SpatialForce;

// TODO(rcory) Break out common code into a test fixture.

/// This is a ForceController integration test. The plant consists of a single
/// planar finger and a fixed brick, and the test checks the ForceController's
/// ability to regulate a fixed force against the surface of the brick.
GTEST_TEST(ForceControllerTest, PlanarFingerStaticForceControl) {
  systems::DiagramBuilder<double> builder;

  SceneGraph<double>* scene_graph;
  MultibodyPlant<double>* plant;
  std::tie(plant, scene_graph) = AddMultibodyPlantSceneGraph(
      &builder, std::make_unique<MultibodyPlant<double>>(1e-3),
      std::make_unique<SceneGraph<double>>());

  // Add the planar_finger model.
  const std::string full_name = FindResourceOrThrow(
      "drake/examples/planar_gripper/models/planar_finger.sdf");
  multibody::Parser(plant, scene_graph).AddModelFromFile(full_name);
  WeldFingerFrame<double>(plant);

  // Adds a single fixed brick (specifically for force control testing).
  auto object_file_name = FindResourceOrThrow(
      "drake/examples/planar_gripper/models/fixed_brick.sdf");
  auto brick_index = multibody::Parser(plant, scene_graph)
                         .AddModelFromFile(object_file_name, "brick");
  const multibody::Frame<double>& brick_base_frame =
      plant->GetFrameByName("brick_base_link", brick_index);
  plant->WeldFrames(plant->world_frame(), brick_base_frame);

  plant->Finalize();

  plant->set_penetration_allowance(0.2);
  plant->set_stiction_tolerance(1e-3);

  lcm::DrakeLcm drake_lcm;
  geometry::ConnectDrakeVisualizer(&builder, *scene_graph, &drake_lcm);
  ConnectContactResultsToDrakeVisualizer(
      &builder, *plant, plant->get_contact_results_output_port());

  // Setup the force control parameters. These values are experimentally known
  // to work well in simulation.
  ForceControlOptions foptions;
  foptions.kpf_t_ = 1e3;
  foptions.kpf_n_ = 5e3;
  foptions.kif_t_ = 1e2;
  foptions.kif_n_ = 1e2;
  foptions.kp_t_ = 0;
  foptions.kd_t_ = 2e3;
  foptions.kp_n_ = 0;
  foptions.kd_n_ = 15e3;
  foptions.Kd_joint_ << 1.0, 0, 0, 1.0;
  foptions.K_compliance_ = 2e3;
  foptions.D_damping_ = 1e3;
  foptions.always_direct_force_control_ = false;
  foptions.finger_to_control_ = Finger::kFinger1;

  // Connect finger/plant states to force controller.
  auto force_controller =
      builder.AddSystem<ForceController>(*plant, *scene_graph, foptions);
  builder.Connect(plant->get_state_output_port(),
                  force_controller->get_finger_state_actual_input_port());
  builder.Connect(plant->get_state_output_port(),
                  force_controller->get_plant_state_actual_input_port());

  // Connect the "virtual" force sensor to the force controller.
  auto zoh_contact_results = builder.AddSystem<systems::ZeroOrderHold<double>>(
      1e-3, Value<ContactResults<double>>());
  builder.Connect(plant->get_contact_results_output_port(),
                  zoh_contact_results->get_input_port());
  std::vector<SpatialForce<double>> init_spatial_vec{
      SpatialForce<double>(Vector3<double>::Zero(), Vector3<double>::Zero())};
  auto zoh_reaction_forces = builder.AddSystem<systems::ZeroOrderHold<double>>(
      1e-3, Value<std::vector<SpatialForce<double>>>(init_spatial_vec));
  builder.Connect(plant->get_reaction_forces_output_port(),
                  zoh_reaction_forces->get_input_port());
  auto force_demux_sys =
      builder.AddSystem<ForceDemuxer>(*plant, foptions.finger_to_control_);
  builder.Connect(zoh_contact_results->get_output_port(),
                  force_demux_sys->get_contact_results_input_port());
  builder.Connect(zoh_reaction_forces->get_output_port(),
                  force_demux_sys->get_reaction_forces_input_port());
  builder.Connect(plant->get_state_output_port(),
                  force_demux_sys->get_state_input_port());
  builder.Connect(force_demux_sys->get_reaction_vec_output_port(),
                  force_controller->get_force_sensor_input_port());

  // Connects the desired fingertip state to the force controller. Currently
  // desired position is unused by the force controller (for now). Additionally,
  // the desired velocity of the contact point is always zero (strict damping),
  // so we set the entire state vector to zero. The 6-element vector represents
  // positions and velocities for the fingertip contact point x-y-z. The
  // controller internally ignores the x-components.
  auto tip_state_desired_src =
      builder.AddSystem<systems::ConstantVectorSource>(Vector6<double>::Zero());
  builder.Connect(tip_state_desired_src->get_output_port(),
                  force_controller->get_contact_state_desired_input_port());

  // Connect the force controller to the plant.
  builder.Connect(force_controller->get_torque_output_port(),
                  plant->get_actuation_input_port());

  // Create a source for the desired force.  The actual force chosen here for
  // this test is arbitrary, with magnitudes on the same scale as what is seen
  // for the brick rotation simulation.
  multibody::ExternallyAppliedSpatialForce<double> desired_force;
  desired_force.F_Bq_W = multibody::SpatialForce<double>(
      Eigen::Vector3d::Zero() /* torque */,
      Eigen::Vector3d(0 /* fx_*/, -0.032 /* fy */, -0.065 /* fz*/));

  // Draw the desired force in drake visualizer. I (rcory) determined the origin
  // of this force vector (p_BoBq_W) experimentally, by rolling out the
  // simulation and analyzing the contact point location after settling. Note:
  // since this test does not control contact position, it can lie anywhere on
  // the brick's surface.
  std::vector<multibody::SpatialForceOutput<double>> force_viz_vec;
  Vector3d p_BoBq_W(0, -0.015819, 0.049337);
  force_viz_vec.emplace_back(p_BoBq_W, desired_force.F_Bq_W);
  auto force_viz_src = builder.AddSystem<systems::ConstantValueSource<double>>(
      Value<std::vector<multibody::SpatialForceOutput<double>>>(
          {force_viz_vec}));
  multibody::ConnectSpatialForcesToDrakeVisualizer(
      &builder, *plant, force_viz_src->get_output_port(0), &drake_lcm);

  // Note, body index and contact point are not (currently) used in the force
  // controller. We set them here anyway.
  desired_force.body_index = plant->GetBodyByName("brick_link").index();
  desired_force.p_BoBq_B = Eigen::Vector3d::Zero();

  auto desired_force_src =
      builder.AddSystem<systems::ConstantValueSource<double>>(
          Value<std::vector<multibody::ExternallyAppliedSpatialForce<double>>>(
              {desired_force}));
  builder.Connect(desired_force_src->get_output_port(0),
                  force_controller->get_force_desired_input_port());

  // Add the system which calculates the brick's contact face & witness/contact
  // point.
  const std::vector<Finger> fingers = {foptions.finger_to_control_};
  auto finger_face_assigner =
      builder.AddSystem<FingerFaceAssigner>(*plant, *scene_graph, fingers);
  builder.Connect(zoh_contact_results->get_output_port(),
                  finger_face_assigner->GetInputPort("contact_results"));
  builder.Connect(scene_graph->get_query_output_port(),
                  finger_face_assigner->GetInputPort("geometry_query"));
  builder.Connect(plant->get_state_output_port(),
                  finger_face_assigner->GetInputPort("plant_state"));
  builder.Connect(
      finger_face_assigner->GetOutputPort("finger_face_assignments"),
      force_controller->GetInputPort("finger_face_assignments"));

  auto diagram = builder.Build();

  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(*plant, diagram_context.get());
  Eigen::VectorXd tau_actuation = Eigen::VectorXd::Zero(2);

  // Set arbitrary initial joint positions.
  plant->SetPositions(&plant_context, Eigen::Vector2d(0.5, -1.4));

  // Add standard gravity in the -z axis.
  plant->mutable_gravity_field().set_gravity_vector(Vector3d(
      0, 0, -multibody::UniformGravityFieldElement<double>::kDefaultStrength));

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_target_realtime_rate(1);
  simulator.Initialize();
  simulator.AdvanceTo(2);  // Run to approximate convergence.

  // Check the steady state actual force against the desired force.
  const auto& post_sim_context = simulator.get_context();
  const auto& post_force_controller_context =
      diagram->GetSubsystemContext(*force_controller, post_sim_context);

  const Eigen::Vector3d F_Bq_W_actual =
      force_controller
          ->EvalVectorInput(
              post_force_controller_context,
              force_controller->get_force_sensor_input_port().get_index())
          ->get_value();
  // Check to within a threshold. This threshold is highly dependent on the
  // force controller gains chosen and how long the simulation is run.
  EXPECT_TRUE(CompareMatrices(desired_force.F_Bq_W.translational(),
                              F_Bq_W_actual, 4.7e-4));
}

// Test at a second finger configuration with a much larger force.
GTEST_TEST(ForceControllerTest, PlanarFingerStaticForceControl2) {
  systems::DiagramBuilder<double> builder;

  SceneGraph<double>* scene_graph;
  MultibodyPlant<double>* plant;
  std::tie(plant, scene_graph) = AddMultibodyPlantSceneGraph(
      &builder, std::make_unique<MultibodyPlant<double>>(1e-3),
      std::make_unique<SceneGraph<double>>());

  // Add the planar_finger model.
  const std::string full_name = FindResourceOrThrow(
      "drake/examples/planar_gripper/models/planar_finger.sdf");
  multibody::Parser(plant, scene_graph).AddModelFromFile(full_name);
  WeldFingerFrame<double>(plant, FingerWeldAngle(Finger::kFinger1));

  // Adds a single fixed brick (specifically for force control testing).
  auto object_file_name = FindResourceOrThrow(
      "drake/examples/planar_gripper/models/fixed_brick.sdf");
  auto brick_index = multibody::Parser(plant, scene_graph)
      .AddModelFromFile(object_file_name, "brick");
  const multibody::Frame<double>& brick_base_frame =
      plant->GetFrameByName("brick_base_link", brick_index);
  math::RigidTransformd weld_xform;
  weld_xform.set_rotation(
      math::RollPitchYaw<double>(0.35880, 0, 0));
  weld_xform.set_translation(
      Eigen::Vector3d(0, -0.00854, 0.03759));
  plant->WeldFrames(plant->world_frame(), brick_base_frame, weld_xform);

  plant->Finalize();

  plant->set_penetration_allowance(0.2);
  plant->set_stiction_tolerance(1e-3);

  lcm::DrakeLcm drake_lcm;
  geometry::ConnectDrakeVisualizer(&builder, *scene_graph, &drake_lcm);
  ConnectContactResultsToDrakeVisualizer(
      &builder, *plant, plant->get_contact_results_output_port());

  // Setup the force control parameters. These values are experimentally known
  // to work well in simulation.
  ForceControlOptions foptions;
  foptions.kpf_t_ = 1.0e3;
  foptions.kpf_n_ = 5e3;
  foptions.kif_t_ = 1e2;
  foptions.kif_n_ = 1e2;
  foptions.kp_t_ = 0;
  foptions.kd_t_ = 2e3;
  foptions.kp_n_ = 0;
  foptions.kd_n_ = 15e3;
  foptions.Kd_joint_ << 1.0, 0, 0, 1.0;
  foptions.K_compliance_ = 2e3;
  foptions.D_damping_ = 1e3;
  foptions.always_direct_force_control_ = false;
  foptions.finger_to_control_ = Finger::kFinger1;

  // Connect finger/plant states to force controller.
  auto force_controller =
      builder.AddSystem<ForceController>(*plant, *scene_graph, foptions);
  builder.Connect(plant->get_state_output_port(),
                  force_controller->get_finger_state_actual_input_port());
  builder.Connect(plant->get_state_output_port(),
                  force_controller->get_plant_state_actual_input_port());

  // Connect the "virtual" force sensor to the force controller.
  auto zoh_contact_results = builder.AddSystem<systems::ZeroOrderHold<double>>(
      1e-3, Value<ContactResults<double>>());
  builder.Connect(plant->get_contact_results_output_port(),
                  zoh_contact_results->get_input_port());
  std::vector<SpatialForce<double>> init_spatial_vec{
      SpatialForce<double>(Vector3<double>::Zero(), Vector3<double>::Zero())};
  auto zoh_reaction_forces = builder.AddSystem<systems::ZeroOrderHold<double>>(
      1e-3, Value<std::vector<SpatialForce<double>>>(init_spatial_vec));
  builder.Connect(plant->get_reaction_forces_output_port(),
                  zoh_reaction_forces->get_input_port());
  auto force_demux_sys =
      builder.AddSystem<ForceDemuxer>(*plant, foptions.finger_to_control_);
  builder.Connect(zoh_contact_results->get_output_port(),
                  force_demux_sys->get_contact_results_input_port());
  builder.Connect(zoh_reaction_forces->get_output_port(),
                  force_demux_sys->get_reaction_forces_input_port());
  builder.Connect(plant->get_state_output_port(),
                  force_demux_sys->get_state_input_port());
  builder.Connect(force_demux_sys->get_reaction_vec_output_port(),
                  force_controller->get_force_sensor_input_port());

  // Connects the desired fingertip state to the force controller. Currently
  // desired position is unused by the force controller (for now). Additionally,
  // the desired velocity of the contact point is always zero (strict damping),
  // so we set the entire state vector to zero. The 6-element vector represents
  // positions and velocities for the fingertip contact point x-y-z. The
  // controller internally ignores the x-components.
  auto tip_state_desired_src =
      builder.AddSystem<systems::ConstantVectorSource>(Vector6<double>::Zero());
  builder.Connect(tip_state_desired_src->get_output_port(),
                  force_controller->get_contact_state_desired_input_port());

  // Connect the force controller to the plant.
  builder.Connect(force_controller->get_torque_output_port(),
                  plant->get_actuation_input_port());

  // Create a source for the desired force.  The actual force chosen here for
  // this test is arbitrary, with magnitudes on the same scale as what is seen
  // for the brick rotation simulation.
  multibody::ExternallyAppliedSpatialForce<double> desired_force;
  desired_force.F_Bq_W = multibody::SpatialForce<double>(
      Eigen::Vector3d::Zero() /* torque */,
      Eigen::Vector3d(0 /* fx_*/, 0.183284 /* fy */, 0.1198 /* fz*/));

  // Draw the desired force in drake visualizer. I (rcory) determined the origin
  // of this force vector (p_BoBq_W) experimentally, by rolling out the
  // simulation and analyzing the contact point location after settling. Note:
  // since this test does not control contact position, it can lie anywhere on
  // the brick's surface.
  std::vector<multibody::SpatialForceOutput<double>> force_viz_vec;
  Vector3d p_BoBq_W(0, -0.05202649, 0.017351387);
  force_viz_vec.emplace_back(p_BoBq_W, desired_force.F_Bq_W);
  auto force_viz_src = builder.AddSystem<systems::ConstantValueSource<double>>(
      Value<std::vector<multibody::SpatialForceOutput<double>>>(
          {force_viz_vec}));
  multibody::ConnectSpatialForcesToDrakeVisualizer(
      &builder, *plant, force_viz_src->get_output_port(0), &drake_lcm);

  // Note, body index and contact point are not (currently) used in the force
  // controller. We set them here anyway.
  desired_force.body_index = plant->GetBodyByName("brick_link").index();
  desired_force.p_BoBq_B = Eigen::Vector3d::Zero();

  auto desired_force_src =
      builder.AddSystem<systems::ConstantValueSource<double>>(
          Value<std::vector<multibody::ExternallyAppliedSpatialForce<double>>>(
              {desired_force}));
  builder.Connect(desired_force_src->get_output_port(0),
                  force_controller->get_force_desired_input_port());

  // Add the system which calculates the brick's contact face & witness/contact
  // point.
  const std::vector<Finger> fingers = {foptions.finger_to_control_};
  auto finger_face_assigner =
      builder.AddSystem<FingerFaceAssigner>(*plant, *scene_graph, fingers);
  builder.Connect(zoh_contact_results->get_output_port(),
                  finger_face_assigner->GetInputPort("contact_results"));
  builder.Connect(scene_graph->get_query_output_port(),
                  finger_face_assigner->GetInputPort("geometry_query"));
  builder.Connect(plant->get_state_output_port(),
                  finger_face_assigner->GetInputPort("plant_state"));
  builder.Connect(
      finger_face_assigner->GetOutputPort("finger_face_assignments"),
      force_controller->GetInputPort("finger_face_assignments"));

  auto diagram = builder.Build();

  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(*plant, diagram_context.get());
  Eigen::VectorXd tau_actuation = Eigen::VectorXd::Zero(2);

  // Set arbitrary initial joint positions.
  plant->SetPositions(&plant_context,
                      Eigen::Vector2d(-0.693555, 1.18458));

  // Add standard gravity in the -z axis.
  plant->mutable_gravity_field().set_gravity_vector(Vector3d(
      0, 0, -multibody::UniformGravityFieldElement<double>::kDefaultStrength));

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_target_realtime_rate(1.1);
  simulator.Initialize();
  simulator.AdvanceTo(2);  // Run to approximate convergence.

  // Check the steady state actual force against the desired force.
  const auto& post_sim_context = simulator.get_context();
  const auto& post_force_controller_context =
      diagram->GetSubsystemContext(*force_controller, post_sim_context);

  const Eigen::Vector3d F_Bq_W_actual =
      force_controller
          ->EvalVectorInput(
              post_force_controller_context,
              force_controller->get_force_sensor_input_port().get_index())
          ->get_value();
  // Check to within a threshold. This threshold is highly dependent on the
  // force controller gains chosen and how long the simulation is run.
  EXPECT_TRUE(CompareMatrices(desired_force.F_Bq_W.translational(),
                              F_Bq_W_actual, 2e-4));
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
