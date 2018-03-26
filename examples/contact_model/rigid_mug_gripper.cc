/*! @file
 A toy example of rubber grips holding a coffee mug.

 The rubber grips are two, symmetric toroidal pieces of rubber. The toroidal
 surface is approximated by uniform sampling its major radius with spheres.
 The number of samples (and many other features of the pads) are configurable
 on the command-line. The gripper is positioned to grab the mug around the
 cylinder's center, assuming the cylinder is in its "identity" pose.

 The mug is modeled as a cylinder with an affixed box approximating the handle.
 The box is *not* a collision element; it merely gives visual representation to
 the mug's asymmetrical inertia properties.

 Ideally, the simulation should show the mug being held in a static grip -- no
 slip.
*/

#include <algorithm>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/lcmt_contact_results_for_viz.hpp"
#include "drake/manipulation/schunk_wsg/schunk_wsg_constants.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/parsers/sdf_parser.h"
#include "drake/multibody/rigid_body_frame.h"
#include "drake/multibody/rigid_body_plant/compliant_contact_model.h"
#include "drake/multibody/rigid_body_plant/contact_results_to_lcm.h"
#include "drake/multibody/rigid_body_plant/drake_visualizer.h"
#include "drake/multibody/rigid_body_plant/rigid_body_plant.h"
#include "drake/multibody/rigid_body_tree.h"
#include "drake/multibody/rigid_body_tree_construction.h"
#include "drake/systems/analysis/runge_kutta3_integrator.h"
#include "drake/systems/analysis/runge_kutta2_integrator.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/util/drakeGeometryUtil.h"
#include "drake/multibody/joints/fixed_joint.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/lcmt_schunk_wsg_status.hpp"
#include "drake/manipulation/schunk_wsg/schunk_wsg_lcm.h"

// Global default contact parameters
DEFINE_double(us, 0.9, "The coefficient of static friction");
DEFINE_double(ud, 0.5, "The coefficient of dynamic friction");
DEFINE_double(youngs_modulus, 1e8, "The contact material Young's modulus (Pa)");
DEFINE_double(dissipation, 2.0, "The contact material dissipation (s/m)");
DEFINE_double(v_stiction_tolerance, 0.01,
              "The maximum slipping speed allowed during stiction (m/s)");
DEFINE_double(contact_radius, 1e-4,
              "The characteristic scale of radius (m) of the contact area");

// Simulation parameters: solver, integrator, playback
DEFINE_double(sim_duration, 5, "Amount of time to simulate (s)");
DEFINE_bool(playback, true,
            "If true, simulation begins looping playback when complete");
DEFINE_string(simulation_type, "compliant", "The type of simulation to use: "
    "'compliant' or 'timestepping'");
DEFINE_string(rk_type, "rk3", "The RK integrator order. Can be 'rk2' or 'rk3'."
    "Used when simulation_time is 'compliant'");
DEFINE_double(ts_dt, 1e-3, "The step size to use for "
    "'simulation_type=timestepping' (ignored for "
    "'simulation_type=compliant'");
DEFINE_double(rk_dt, 1e-4, "The step size to use for "
    "'simulation_type=compliant' (ignored for "
    "'simulation_type=timestepping'");
DEFINE_double(accuracy, 5e-5, "Sets the simulation accuracy for "
    "'simulation_type=compliant'");
DEFINE_bool(print_time, false, "Prints simulation timestamp every 1/10 s.");

// Parameters for specifying the ring pad approximation.
DEFINE_int32(ring_samples, 4,
             "The number of spheres used to sample the pad ring");
DEFINE_double(pad_depth, 4e-3, "The depth the foremost pads penetrate the mug. "
    "Deeper penetration implies stronger contact forces");
DEFINE_double(ring_orient, 0, "Rotation of pads around x-axis (in degrees)");
DEFINE_double(ring_youngs_modulus, -1, "The Young's modulus for the ring pad. "
    "Negative values use the global default");
DEFINE_double(ring_dissipation, -1, "The dissipation for the ring pad. "
    "Negative values use the global default");
DEFINE_double(ring_static_friction, -1, "The coefficient of static friction for the ring pad. "
    "Negative values use the global default");
DEFINE_double(ring_dynamic_friction, -1, "The coefficient of dynamic friction for the ring pad."
    " Negative values use the global default");

// Parameters for posing the mug.
DEFINE_double(px, 0, "The x-position of the center, bottom of the mug");
DEFINE_double(py, 0, "The y-position of the center, bottom of the mug");
DEFINE_double(pz, 0, "The z-position of the center, bottom of the mug");
DEFINE_double(rx, 0,
              "The x-rotation of the mug around its origin - the center of its "
                  "bottom (in degrees). Rotation order: X, Y, Z");
DEFINE_double(ry, 0,
              "The y-rotation of the mug around its origin - the center of its "
                  "bottom (in degrees). Rotation order: X, Y, Z");
DEFINE_double(rz, 0,
              "The z-rotation of the mug around its origin - the center of its "
                  "bottom (in degrees). Rotation order: X, Y, Z");

DEFINE_double(gripper_force, 0, "The force to be applied by the gripper. A value"
              "of 0 indicates no gripper (uses pad_depth to determine"
              " penetration distance).");

namespace drake {
namespace examples {

using drake::SquareTwistMatrix;
using drake::systems::RungeKutta3Integrator;
using drake::systems::RungeKutta2Integrator;
using drake::systems::ContactResultsToLcmSystem;
using drake::systems::lcm::LcmPublisherSystem;
using drake::trajectories::PiecewisePolynomial;
using drake::manipulation::schunk_wsg::SchunkWsgStatusSender;
using Eigen::Matrix3d;
using std::make_unique;

// These values should match the cylinder defined in:
// drake/examples/contact_model/cylinder_mug.urdf
const double kMugHeight = 0.1;
const double kMugRadius = 0.04;
// The pad was measured as a torus with the following major and minor radii.
const double kPadMajorRadius = 14e-3; // 14 mm.
const double kPadMinorRadius = 6e-3;  // 6 mm.


void AddPads(RigidBodyTree<double>* tree, RigidBody<double>* parent_body,
             const double pad_offset, std::string group_name) {
  const int sample_count = FLAGS_ring_samples;
  const double sample_rotation = FLAGS_ring_orient * M_PI / 180.0; // in radians

  drake::systems::CompliantMaterial material;
  if (FLAGS_ring_youngs_modulus >= 0)
    material.set_youngs_modulus(FLAGS_ring_youngs_modulus);
  if (FLAGS_ring_dissipation >= 0)
    material.set_dissipation(FLAGS_ring_dissipation);
  if (FLAGS_ring_static_friction >= 0 && FLAGS_ring_dynamic_friction >= 0) {
    material.set_friction(FLAGS_ring_static_friction,
                          FLAGS_ring_dynamic_friction);
  } else if (FLAGS_ring_static_friction >= 0 ||
      FLAGS_ring_dynamic_friction >= 0) {
    drake::log()->warn("Both static and dynamic friction should be specified. "
                           "Using global values instead.");
  }
  std::cout << "Ring contact material\n";
  std::cout << "  Youngs modulus:   " << material.youngs_modulus() << "\n";
  std::cout << "  Dissipation:      " << material.dissipation() << "\n";
  std::cout << "  Static friction:  " << material.static_friction() << "\n";
  std::cout << "  Dynamic friction: " << material.dynamic_friction() << "\n";

  // Sample the torus with a sphere. The sphere is located at X_GS, relative to
  // the gripper. X_PS is transform of sphere w.r.t. the parent.
  auto add_sphere =
      [&parent_body, &tree, &material] (auto X_PS, auto sphere_name) {
    // Create the sphere visual geometry
    const DrakeShapes::Sphere sphere(kPadMinorRadius);
    DrakeShapes::VisualElement vis(Eigen::Isometry3d::Identity());
    vis.setGeometry(sphere);
    vis.setMaterial(Eigen::Vector4d(1, 0, 0, 1));

    // Create another body that is welded to the parent
    std::unique_ptr<RigidBody<double>> sphere_body = parent_body->Clone();
    sphere_body->set_name(sphere_name);
    auto joint_name = sphere_name + "_joint";
    sphere_body->add_joint(
        parent_body, std::make_unique<FixedJoint>(joint_name, X_PS));

    // Add the body to the tree and assign the sphere visual element
    auto mbody = tree->add_rigid_body(std::move(sphere_body));
    mbody->AddVisualElement(vis);

    // Add the collision element geometry
    drake::multibody::collision::Element collide(
        Eigen::Isometry3d::Identity(), mbody);
    collide.setGeometry(sphere);
    collide.set_compliant_material(material);
    tree->addCollisionElement(collide, *mbody, "default");
  };

  const double d_theta = 2 * M_PI / sample_count;
  double x_coordinate, y_coordinate, z_coordinate;
  for (int i = 0; i < sample_count; ++i) {
    if (FLAGS_gripper_force > 0) {
      x_coordinate = pad_offset;
      y_coordinate =
          std::cos(d_theta * i + sample_rotation) * kPadMajorRadius + 0.0265;
    } else {
      y_coordinate = pad_offset;
      x_coordinate =
          -std::cos(d_theta * i + sample_rotation) * kPadMajorRadius;
    }
    z_coordinate =
        std::sin(d_theta * i + sample_rotation) * kPadMajorRadius;

    add_sphere(drake::Isometry3<double>{drake::Translation3<double>
                   {x_coordinate, y_coordinate, z_coordinate}},
               "sphere_" + std::to_string(i) + "_" + group_name);
  }
}

std::unique_ptr<RigidBodyTreed> BuildTestTree(int* gripper_instance_id = nullptr) {
  std::unique_ptr<RigidBodyTreed> tree = std::make_unique<RigidBodyTreed>();

  std::string collision_filter_name;
  if (FLAGS_gripper_force == 0) {
    // Add the gripper.  Move it up so it's aligned with the center of the mug's
    // barrel. NOTE: This urdf is an "empty" body. It has mass but no geometry
    // (collision or visual). We add them procedurally.
    auto gripper_frame = std::allocate_shared<RigidBodyFrame<double>>(
        Eigen::aligned_allocator<RigidBodyFrame<double>>(), "gripper_pose_frame",
        &tree->world(), Eigen::Vector3d(0, 0, kMugHeight / 2),
        Eigen::Vector3d::Zero());

    auto gripper_id_table = parsers::urdf::AddModelInstanceFromUrdfFile(
        FindResourceOrThrow("drake/examples/contact_model/rigid_mug_gripper.urdf"),
        multibody::joints::kFixed, gripper_frame, false, tree.get());
    *gripper_instance_id = gripper_id_table.begin()->second;

    // Add the procedural gripper pads.
    RigidBody<double>& gripper_body = *tree->FindBody("gripper_pads", "gripper");
    //AddGripperPads(tree.get(), &gripper_body);

    AddPads(tree.get(), &gripper_body,
            kMugRadius + kPadMinorRadius - FLAGS_pad_depth, "pos_y");
    AddPads(tree.get(), &gripper_body,
            -(kMugRadius + kPadMinorRadius - FLAGS_pad_depth), "neg_y");

    collision_filter_name = "pads_filter";
  } else {
    // Add the gripper.  Move it up so it's aligned with the center of the mug's
    // barrel. NOTE: This urdf is an "empty" body. It has mass but no geometry
    // (collision or visual). We add them procedurally.
    auto gripper_frame = std::allocate_shared<RigidBodyFrame<double>>(
        Eigen::aligned_allocator<RigidBodyFrame<double>>(), "gripper_pose_frame",
        &tree->world(), Eigen::Vector3d(0.055 + 5e-4, 0, kMugHeight / 2 + 5e-4),
        Eigen::Vector3d(0, 0, 1.57));

    auto gripper_id_table = parsers::sdf::AddModelInstancesFromSdfFile(
        FindResourceOrThrow("drake/examples/contact_model/"
                            "schunk_wsg_50_ball_contact_visual_collision.sdf"),
        multibody::joints::kFixed, gripper_frame, false, tree.get());
    *gripper_instance_id = gripper_id_table.begin()->second;

    // Add the procedural gripper pads.
    RigidBody<double>& right_finger_body = *tree->FindBody("right_finger");
    AddPads(tree.get(), &right_finger_body, -0.0046, "pos_y");

    // Add the procedural gripper pads.
    RigidBody<double>& left_finger_body = *tree->FindBody("left_finger");
    AddPads(tree.get(), &left_finger_body, 0.0046, "neg_y");

    collision_filter_name = "finger_filter";
  }

  // Create the pads collision filter.
  for (int i = 0; i < FLAGS_ring_samples; i++) {
    tree->AddCollisionFilterGroupMember(
        collision_filter_name, "sphere_" + std::to_string(i) + "_pos_y",
        *gripper_instance_id);
    tree->AddCollisionFilterGroupMember(
        collision_filter_name, "sphere_" + std::to_string(i) + "_neg_y",
        *gripper_instance_id);
  }
  tree->compile();

  // Add the "Mug" to grip. It assumes that with the identity pose, the center
  // of the bottom of the mug sits on the origin with the rest of the mug
  // oriented in the +z direction.
  std::cout << "Mug pose:\n";
  std::cout << "  Position:    " << FLAGS_px << ", " << FLAGS_py << ", "
            << FLAGS_pz << "\n";
  std::cout << "  Orientation: " << FLAGS_rx << ", " << FLAGS_ry << ", "
            << FLAGS_rz << "\n";
  auto mug_frame = std::allocate_shared<RigidBodyFrame<double>>(
      Eigen::aligned_allocator<RigidBodyFrame<double>>(), "mug_pose",
      &tree->world(), Eigen::Vector3d(FLAGS_px, FLAGS_py, FLAGS_pz),
      Eigen::Vector3d(FLAGS_rx * M_PI / 180, FLAGS_ry * M_PI / 180,
                      (FLAGS_rz * M_PI / 180) + M_PI));
  parsers::urdf::AddModelInstanceFromUrdfFile(
      FindResourceOrThrow("drake/examples/contact_model/cylinder_mug.urdf"),
      multibody::joints::kQuaternion, mug_frame, tree.get());

  return tree;
}

int main() {
  systems::DiagramBuilder<double> builder;

  if (FLAGS_simulation_type == "compliant") {
    FLAGS_ts_dt = 0.0;  // Set the time-stepping dt to zero
    std::cout << "Simulation type:\n";
    std::cout<<"\tcompliant: " << FLAGS_rk_type << "\n";
    std::cout<<"\tdt:        " << FLAGS_rk_dt << "\n";
  }
  else if (FLAGS_simulation_type == "timestepping") {
    std::cout << "Simulation type:\n";
    std::cout << "\ttime-stepping  \n";
    std::cout << "\tdt:              " << FLAGS_ts_dt << "\n";
  } else {
    throw std::runtime_error(
        "Simulation type" + FLAGS_simulation_type + "is not supported");
  }

  // Set the gripper force source.
  int gripper_instance_id = -1;
  systems::RigidBodyPlant<double>* plant;
  if (FLAGS_gripper_force == 0) {
    plant =
        builder.AddSystem<systems::RigidBodyPlant<double>>(
            BuildTestTree(&gripper_instance_id), FLAGS_ts_dt);
  } else {
    plant =
        builder.AddSystem<systems::RigidBodyPlant<double>>(
            BuildTestTree(&gripper_instance_id), FLAGS_ts_dt);
    // Create a trajectory for grip force.
    // Settle the grip by the time the lift starts.
    std::vector<double> grip_breaks{0., 0.01, FLAGS_sim_duration};
    std::vector<Eigen::MatrixXd> grip_knots;
    grip_knots.push_back(Vector1d(0));
    grip_knots.push_back(Vector1d(FLAGS_gripper_force));
    grip_knots.push_back(Vector1d(FLAGS_gripper_force));
    PiecewisePolynomial<double> grip_trajectory =
        PiecewisePolynomial<double>::FirstOrderHold(grip_breaks, grip_knots);
    auto grip_source =
        builder.AddSystem<systems::TrajectorySource>(grip_trajectory);
    grip_source->set_name("grip_source");
    builder.Connect(grip_source->get_output_port(),
                    plant->model_instance_actuator_command_input_port(
                        gripper_instance_id));
  }
  plant->set_name("plant");

  // Command-line specified contact parameters.
  std::cout << "Contact properties:\n";
  std::cout << "  Young's modulus:          " << FLAGS_youngs_modulus << "\n";
  std::cout << "  Dissipation:              " << FLAGS_dissipation << "\n";
  std::cout << "  Penetration depth:        " << FLAGS_pad_depth << "\n";
  std::cout << "  static friction:          " << FLAGS_us << "\n";
  std::cout << "  dynamic friction:         " << FLAGS_ud << "\n";
  std::cout << "  Allowed stiction speed:   " << FLAGS_v_stiction_tolerance
            << "\n";

  systems::CompliantMaterial default_material;
  default_material.set_youngs_modulus(FLAGS_youngs_modulus)
      .set_dissipation(FLAGS_dissipation)
      .set_friction(FLAGS_us, FLAGS_ud);
  plant->set_default_compliant_material(default_material);
  systems::CompliantContactModelParameters model_parameters;
  model_parameters.characteristic_radius = FLAGS_contact_radius;
  model_parameters.v_stiction_tolerance = FLAGS_v_stiction_tolerance;
  plant->set_contact_model_parameters(model_parameters);

  // Creates and adds LCM publisher for visualization.  The test doesn't
  // require `drake_visualizer` but it is convenient to have when debugging.
  drake::lcm::DrakeLcm lcm;
  const auto viz_publisher =
      builder.template AddSystem<systems::DrakeVisualizer>(
          plant->get_rigid_body_tree(), &lcm, true);
  builder.Connect(plant->state_output_port(),
                  viz_publisher->get_input_port(0));

  // Enable contact force visualization.
  const ContactResultsToLcmSystem<double>& contact_viz =
      *builder.template AddSystem<ContactResultsToLcmSystem<double>>(
          plant->get_rigid_body_tree());
  auto& contact_results_publisher = *builder.AddSystem(
      LcmPublisherSystem::Make<lcmt_contact_results_for_viz>(
          "CONTACT_RESULTS", &lcm));
  // Contact results to lcm msg.
  builder.Connect(plant->contact_results_output_port(),
                  contact_viz.get_input_port(0));
  builder.Connect(contact_viz.get_output_port(0),
                  contact_results_publisher.get_input_port());

  if (FLAGS_gripper_force > 0) {
    // Create the wsg status publisher.
    auto wsg_status_pub =
        builder.AddSystem(
            LcmPublisherSystem::Make<drake::lcmt_schunk_wsg_status>(
                "SCHUNK_WSG_STATUS", &lcm));
    wsg_status_pub->set_name("wsg_status_publisher");
    wsg_status_pub->set_publish_period(
        drake::manipulation::schunk_wsg::kSchunkWsgLcmStatusPeriod);

    // Get the state and torque output ports for publishing via LCM.
    const auto& wsg_state_output_port =
        plant->model_instance_state_output_port(gripper_instance_id);
//    auto wsg_state_output_port_indx =
//        builder.ExportOutput(wsg_state_output_port);

    const auto& wsg_torque_output_port =
        plant->model_instance_torque_output_port(gripper_instance_id);
//    auto wsg_torque_output_port_indx =
//        builder.ExportOutput(wsg_torque_output_port);

    // Create the status sender.
    auto wsg_status_sender =
        builder.AddSystem<drake::manipulation::schunk_wsg::SchunkWsgStatusSender>(
            wsg_state_output_port.size(),
            wsg_torque_output_port.size(),
            drake::manipulation::schunk_wsg::kSchunkWsgPositionIndex,
            drake::manipulation::schunk_wsg::kSchunkWsgVelocityIndex);
    wsg_status_sender->set_name("wsg_status_sender");

    builder.Connect(wsg_state_output_port,
                    wsg_status_sender->get_input_port_wsg_state());
    builder.Connect(wsg_torque_output_port,
                    wsg_status_sender->get_input_port_measured_torque());
    builder.Connect(*wsg_status_sender, *wsg_status_pub);
  }

  // Build the diagram.
  const std::unique_ptr<systems::Diagram<double>> model = builder.Build();

  // Set up the model and simulator and set their starting state.
  systems::Simulator<double> simulator(*model);

  systems::Context<double>& context = simulator.get_mutable_context();

  if (FLAGS_gripper_force > 0) {
    // Open the gripper.
    plant->SetModelInstancePositions(
        &model->GetMutableSubsystemContext(
            *plant, &context), gripper_instance_id,
        manipulation::schunk_wsg::GetSchunkWsgMugGraspPosition<double>());
  }

  if (FLAGS_rk_type == "rk2") {
    simulator.reset_integrator<RungeKutta2Integrator<double>>(
        *model, FLAGS_rk_dt, &simulator.get_mutable_context());
  } else if (FLAGS_rk_type == "rk3") {
    simulator.reset_integrator<RungeKutta3Integrator<double>>(*model, &context);
    simulator.get_mutable_integrator()->
        request_initial_step_size_target(FLAGS_rk_dt);
    simulator.get_mutable_integrator()->set_target_accuracy(FLAGS_accuracy);
    std::cout << "Variable-step integrator accuracy: " << FLAGS_accuracy << "\n";
  } else {
    throw std::runtime_error("RK type not recognized.");
  }

  simulator.Initialize();

  if (FLAGS_print_time) {
    // Print a time stamp update every tenth of a second.  This helps communicate
    // progress in the event that the integrator crawls to a very small timestep.
    const double kPrintPeriod = std::min(0.1, FLAGS_sim_duration);
    int step_count =
        static_cast<int>(std::ceil(FLAGS_sim_duration / kPrintPeriod));
    for (int i = 1; i <= step_count; ++i) {
      double t = context.get_time();
      std::cout << "time: " << t << "\n";
      simulator.StepTo(i * kPrintPeriod);
    }
  } else {
    simulator.StepTo(FLAGS_sim_duration);
  }

  while (FLAGS_playback) viz_publisher->ReplayCachedSimulation();
  return 0;
}

}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::main();
}
