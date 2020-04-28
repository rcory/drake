#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/examples/planar_gripper/finger_brick.h"
#include "drake/examples/planar_gripper/finger_brick_control.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_value_source.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/zero_order_hold.h"

namespace drake {
namespace examples {
namespace planar_gripper {

using multibody::MultibodyPlant;
using multibody::AddMultibodyPlantSceneGraph;
using geometry::SceneGraph;
using multibody::ContactResults;
using multibody::SpatialForce;

/// This is a ForceController integration test. The plant consists of a single
/// planar finger and a fixed brick, and the test checks the ForceController's
/// ability to regulate a fixed force against the surface of the brick.
GTEST_TEST(ForceControllerTest, PlanarFingerStaticForceControl) {
  systems::DiagramBuilder<double> builder;

  SceneGraph<double>* scene_graph;
  MultibodyPlant<double>* plant;
  std::tie(plant, scene_graph) = AddMultibodyPlantSceneGraph(
      &builder, std::make_unique<MultibodyPlant<double>>(1e-3),
      std::make_unique <SceneGraph<double>>());

  // Add the planar_finger model.
  const std::string full_name =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_finger.sdf");
  multibody::Parser(plant, scene_graph).AddModelFromFile(full_name);
  WeldFingerFrame<double>(plant);

  // Adds a single fixed brick (specifically for force control testing).
  auto object_file_name =
      FindResourceOrThrow("drake/examples/planar_gripper/fixed_brick.sdf");
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
  foptions.kpf_t_ = 3e3;
  foptions.kpf_n_ = 5e3;
  foptions.kif_t_ = 1e2;
  foptions.kif_n_ = 1e2;
  foptions.kp_t_ = 0;
  foptions.kd_t_ = 20e2;
  foptions.kp_n_ = 0;
  foptions.kd_n_ = 15e3;
  foptions.Kd_joint_ << 1.0, 0, 0, 1.0;
  foptions.K_compliance_ = 2e3;
  foptions.D_damping_ = 1e3;
  foptions.always_direct_force_control_ = false;
  foptions.finger_to_control_ = Finger::kFinger1;
  foptions.brick_face_ = BrickFace::kPosZ;

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
      builder.AddSystem<ForceDemuxer>(*plant, Finger::kFinger1);
  builder.Connect(zoh_contact_results->get_output_port(),
                   force_demux_sys->get_contact_results_input_port());
  builder.Connect(zoh_reaction_forces->get_output_port(),
                   force_demux_sys->get_reaction_forces_input_port());
  builder.Connect(plant->get_state_output_port(),
                  force_demux_sys->get_state_input_port());
  builder.Connect(force_demux_sys->get_reaction_vec_output_port(),
                  force_controller->get_force_sensor_input_port());

  // Connects the desired state to force controller (currently desired state is
  // unused by the force controller). Since we currently don't regulate position
  // for now, set these to zero. 6-vector represents pos-vel for fingertip
  // contact point x-y-z. The controller ignores the x-components.
  auto tip_state_desired_src =
      builder.AddSystem<systems::ConstantVectorSource>(Vector6<double>::Zero());
  builder.Connect(tip_state_desired_src->get_output_port(),
                   force_controller->get_contact_state_desired_input_port());

  // Connect the force controller to the plant.
  builder.Connect(force_controller->get_torque_output_port(),
                  plant->get_actuation_input_port());

  // Create a source for the desired force.  The actual force chosen is
  // arbitrary, but on the same scale as what is produced for the brick rotation
  // simulation.
  multibody::ExternallyAppliedSpatialForce<double> desired_force;
  desired_force.F_Bq_W = multibody::SpatialForce<double>(
      Eigen::Vector3d::Zero() /* torque */,
      Eigen::Vector3d(0 /* fx_*/, -0.032 /* fy */, -0.065 /* fz*/));

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

  // Add the system which calculates the contact point.
  auto contact_point_calc_sys = builder.AddSystem<ContactPointInBrickFrame>(
      *plant, *scene_graph, Finger::kFinger1);
  builder.Connect(zoh_contact_results->get_output_port(),
                  contact_point_calc_sys->GetInputPort("contact_results"));
  builder.Connect(plant->get_state_output_port(),
                  contact_point_calc_sys->GetInputPort("x"));
  builder.Connect(contact_point_calc_sys->GetOutputPort("p_BrCb"),
                  force_controller->get_p_BrCb_input_port());
  builder.Connect(contact_point_calc_sys->GetOutputPort("b_in_contact"),
                  force_controller->get_is_contact_input_port());
  builder.Connect(scene_graph->get_query_output_port(),
                  contact_point_calc_sys->get_geometry_query_input_port());

  auto diagram = builder.Build();

  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(*plant, diagram_context.get());
  Eigen::VectorXd tau_actuation = Eigen::VectorXd::Zero(2);

  // Set arbitrary initial joint positions.
  plant->SetPositions(&plant_context, Eigen::Vector2d(0.5, -1.4));

  // No gravity.
  plant->mutable_gravity_field().set_gravity_vector(Eigen::Vector3d::Zero());

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_target_realtime_rate(1);
  simulator.Initialize();
  simulator.AdvanceTo(1.2);

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
  // Check to within an arbitrary threshold.
  drake::log()->info("F_Bq_W_actual: \n{}", F_Bq_W_actual.transpose());
  drake::log()->info("F_Bq_W_desired: \n{}",
                     desired_force.F_Bq_W.translational().transpose());
  drake::log()->info("delta: \n{}",
                     desired_force.F_Bq_W.translational().transpose() -
                         F_Bq_W_actual.transpose());
  EXPECT_TRUE(CompareMatrices(desired_force.F_Bq_W.translational(),
                              F_Bq_W_actual, 1.9e-4));
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake