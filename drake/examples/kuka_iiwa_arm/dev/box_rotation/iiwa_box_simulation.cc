/// @file
///
/// Implements a simulation of the KUKA iiwa arm with a Schunk WSG 50
/// attached as an end effector.  Like the driver for the physical
/// arm, this simulation communicates over LCM using lcmt_iiwa_status
/// and lcmt_iiwa_command messages for the arm, and the
/// lcmt_schunk_status and lcmt_schunk_command messages for the
/// gripper. It is intended to be a be a direct replacement for the
/// KUKA iiwa driver and the actual robot hardware.

#include <memory>

#include <gflags/gflags.h>

#include "drake/examples/kuka_iiwa_arm/iiwa_common.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_lcm.h"
#include "drake/examples/kuka_iiwa_arm/dev/box_rotation/iiwa_box_diagram_factory.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_world/world_sim_tree_builder.h"
#include "drake/examples/kuka_iiwa_arm/oracular_state_estimator.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/lcmt_iiwa_command.hpp"
#include "drake/lcmt_iiwa_status.hpp"
#include "drake/lcmtypes/drake/lcmt_contact_results_for_viz.hpp"
#include "drake/multibody/rigid_body_plant/contact_results_to_lcm.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_plant/drake_visualizer.h"
#include "drake/multibody/rigid_body_plant/rigid_body_plant.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/runge_kutta2_integrator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/matrix_gain.h"
#include "drake/util/drakeGeometryUtil.h"

DEFINE_double(simulation_sec, std::numeric_limits<double>::infinity(),
              "Number of seconds to simulate.");

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace box_rotation {
namespace {

using systems::Context;
using systems::Diagram;
using systems::DiagramBuilder;
using systems::DrakeVisualizer;
using systems::InputPortDescriptor;
using systems::OutputPort;
using systems::RigidBodyPlant;
using systems::Simulator;

const char *const kIiwaUrdf =
    "drake/manipulation/models/iiwa_description/urdf/"
        "dual_iiwa14_primitive_collision.urdf";

// TODO(naveen): refactor this to reduce duplicate code.
template<typename T>
std::unique_ptr<RigidBodyPlant<T>> BuildCombinedPlant(
    ModelInstanceInfo<T> *iiwa_instance, ModelInstanceInfo<T> *box_instance) {
  auto tree_builder = std::make_unique<WorldSimTreeBuilder<double>>();

  // Adds models to the simulation builder. Instances of these models can be
  // subsequently added to the world.
  tree_builder->StoreModel("iiwa", kIiwaUrdf);
  tree_builder->StoreModel("table",
                           "drake/examples/kuka_iiwa_arm/models/table/"
                               "extra_heavy_duty_table_surface_only_collision.sdf");
  tree_builder->StoreModel("large_table",
                           "drake/examples/kuka_iiwa_arm/models/table/"
                               "large_extra_heavy_duty_table_surface_only_collision.sdf");
  tree_builder->StoreModel("box",
                           "drake/examples/kuka_iiwa_arm/dev/box_rotation/"
                               "box.urdf");

  // Build a world with two fixed tables.  A box is placed one on
  // table, and the iiwa arm is fixed to the other.
//  tree_builder->AddFixedModelInstance("table", /* right arm */
//                                      Eigen::Vector3d::Zero() /* xyz */,
//                                      Eigen::Vector3d::Zero() /* rpy */);
//  tree_builder->AddFixedModelInstance("table", /* left arm */
//                                      Eigen::Vector3d(0, 0.96, 0) /* xyz */,
//                                      Eigen::Vector3d::Zero() /* rpy */);
//  tree_builder->AddFixedModelInstance("large_table", /* box */
//                                      Eigen::Vector3d(0.72, 0.96/2,0) /* xyz */,
//                                      Eigen::Vector3d::Zero() /* rpy */);

  tree_builder->AddGround();

  // The `z` coordinate of the top of the table in the world frame.
  // The quantity 0.736 is the `z` coordinate of the frame associated with the
  // 'surface' collision element in the SDF. This element uses a box of height
  // 0.057m thus giving the surface height (`z`) in world coordinates as
  // 0.736 + 0.057 / 2.
  const double kTableTopZInWorld = 0; //0.736 + 0.057 / 2;

  // Coordinates for kRobotBase originally from iiwa_world_demo.cc.
  // The intention is to center the robot on the table.
  const Eigen::Vector3d kRobotBase(0, 0, kTableTopZInWorld);
  // Start the box slightly above the table.  If we place it at
  // the table top exactly, it may start colliding the table (which is
  // not good, as it will likely shoot off into space).
  const Eigen::Vector3d kBoxBase(0.7, 0.96/2 , kTableTopZInWorld + 0.56/2);

  int id = tree_builder->AddFixedModelInstance("iiwa", kRobotBase);
  *iiwa_instance = tree_builder->get_model_info_for_instance(id);
  id = tree_builder->AddFloatingModelInstance("box", kBoxBase,
                                              Vector3<double>(0, 0, 0));
  *box_instance = tree_builder->get_model_info_for_instance(id);

  auto plant = std::make_unique<RigidBodyPlant<T>>(tree_builder->Build());

  return plant;
}

int DoMain() {
  systems::DiagramBuilder<double> builder;

  ModelInstanceInfo<double> iiwa_instance, box_instance;

  std::unique_ptr<systems::RigidBodyPlant<double>> model_ptr =
      BuildCombinedPlant<double>(&iiwa_instance, &box_instance);
  model_ptr->set_name("plant");

  // Arbitrary contact parameters.
  const double kStiffness = 3000;
  const double kDissipation = 5;
  const double kStaticFriction = 10;
  const double kDynamicFriction = 1;
  const double kVStictionTolerance = 0.1;
  model_ptr->set_normal_contact_parameters(kStiffness, kDissipation);
  model_ptr->set_friction_contact_parameters(kStaticFriction, kDynamicFriction,
                                         kVStictionTolerance);

  auto model =
      builder.template AddSystem<IiwaAndBoxPlantWithStateEstimator<double>>(
          std::move(model_ptr), iiwa_instance, box_instance);
  model->set_name("plant_with_state_estimator");

  const RigidBodyTree<double> &tree = model->get_plant().get_rigid_body_tree();

  drake::lcm::DrakeLcm lcm;
  DrakeVisualizer *visualizer = builder.AddSystem<DrakeVisualizer>(tree, &lcm);
  visualizer->set_name("visualizer");
  builder.Connect(model->get_output_port_plant_state(),
                  visualizer->get_input_port(0));
  visualizer->set_publish_period(kIiwaLcmStatusPeriod);

  // Create the command subscriber and status publisher.
  auto iiwa_command_sub = builder.AddSystem(
      systems::lcm::LcmSubscriberSystem::Make<lcmt_iiwa_command>("IIWA_COMMAND",
                                                                 &lcm));
  iiwa_command_sub->set_name("iiwa_command_subscriber");
  auto iiwa_command_receiver = builder.AddSystem<IiwaCommandReceiver>(14);
  iiwa_command_receiver->set_name("iwwa_command_receiver");

  auto iiwa_status_pub = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<lcmt_iiwa_status>("IIWA_STATUS",
                                                               &lcm));
  iiwa_status_pub->set_name("iiwa_status_publisher");
  iiwa_status_pub->set_publish_period(kIiwaLcmStatusPeriod);
  auto iiwa_status_sender = builder.AddSystem<IiwaStatusSender>(14);
  iiwa_status_sender->set_name("iiwa_status_sender");

  // TODO(siyuan): Connect this to kuka_planner runner once it generates
  // reference acceleration.
  auto iiwa_zero_acceleration_source =
      builder.template AddSystem<systems::ConstantVectorSource<double>>(
          Eigen::VectorXd::Zero(14));
  iiwa_zero_acceleration_source->set_name("zero_acceleration");

  builder.Connect(iiwa_command_sub->get_output_port(0),
                  iiwa_command_receiver->get_input_port(0));

  builder.Connect(iiwa_command_receiver->get_output_port(0),
                  model->get_input_port_iiwa_state_command());

  builder.Connect(iiwa_zero_acceleration_source->get_output_port(),
                  model->get_input_port_iiwa_acceleration_command());

//  std::cout<<"output: "<<iiwa_zero_acceleration_source->get_output_port().size()<<std::endl;
//  std::cout<<"input: "<<model->get_input_port_iiwa_acceleration_command().size()<<std::endl;

  builder.Connect(model->get_output_port_iiwa_state(),
                  iiwa_status_sender->get_state_input_port());

  builder.Connect(iiwa_command_receiver->get_output_port(0),
                  iiwa_status_sender->get_command_input_port());

  builder.Connect(iiwa_status_sender->get_output_port(0),
                  iiwa_status_pub->get_input_port(0));

  auto iiwa_state_pub = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<bot_core::robot_state_t>(
          "IIWA_STATE_EST", &lcm));
  iiwa_state_pub->set_name("iiwa_state_publisher");
  iiwa_state_pub->set_publish_period(kIiwaLcmStatusPeriod);

  builder.Connect(model->get_output_port_iiwa_robot_state_msg(),
                  iiwa_state_pub->get_input_port(0));
  iiwa_state_pub->set_publish_period(kIiwaLcmStatusPeriod);

  auto box_state_pub = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<bot_core::robot_state_t>(
          "OBJECT_STATE_EST", &lcm));
  box_state_pub->set_name("box_state_publisher");
  box_state_pub->set_publish_period(kIiwaLcmStatusPeriod);

  builder.Connect(model->get_output_port_box_robot_state_msg(),
                  box_state_pub->get_input_port(0));
  box_state_pub->set_publish_period(kIiwaLcmStatusPeriod);


//// Add contact viz.
  auto contact_viz =
      builder.template AddSystem<systems::ContactResultsToLcmSystem<double>>(
          model->get_tree());
  contact_viz->set_name("contact_viz");

  auto contact_results_publisher = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<lcmt_contact_results_for_viz>(
          "CONTACT_RESULTS", &lcm));
  contact_results_publisher->set_name("contact_results_publisher");

  builder.Connect(model->get_output_port_contact_results(),
                           contact_viz->get_input_port(0));
  builder.Connect(contact_viz->get_output_port(0),
                           contact_results_publisher->get_input_port(0));
  contact_results_publisher->set_publish_period(.01);

  auto sys = builder.Build();
  Simulator<double> simulator(*sys);

  simulator.reset_integrator<systems::RungeKutta2Integrator<double>>(*sys, 1e-3, simulator.get_mutable_context());

  lcm.StartReceiveThread();
  simulator.set_target_realtime_rate(1);
  simulator.Initialize();
  simulator.set_publish_every_time_step(false);
  simulator.StepTo(FLAGS_simulation_sec);
  //simulator.StepTo(.01);

  return 0;
}

}  // namespace
}  // namespace box_rotation
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::kuka_iiwa_arm::box_rotation::DoMain();
}
