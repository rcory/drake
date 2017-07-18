/// @file
///
/// This test sets up a simple passive dynamics simulation of the  mobile
/// robot, i.e., all joint torques are set to zero.

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/common/text_logging_gflags.h"
#include "drake/examples/TRI_Remy/remy_common.h"
#include "drake/examples/TRI_Remy/controller/remy_controller.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/rigid_body_plant/drake_visualizer.h"
#include "drake/multibody/rigid_body_plant/rigid_body_plant.h"
#include "drake/multibody/rigid_body_tree_construction.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/multibody/joints/drake_joints.h"

DEFINE_double(stiffness,10000,"stiffness");
DEFINE_double(dissipation,2,"dissipation");
DEFINE_double(z0,0.12,"initial base height");

DEFINE_double(bx,-0.3,"blockx");
DEFINE_double(by,-1.3,"blocky");
DEFINE_double(bz,0.5,"blockz");

namespace drake {

using systems::Context;
using systems::ContinuousState;
using systems::RigidBodyPlant;
using systems::VectorBase;

namespace examples {
namespace Remy {

const char* kRelUrdfPath =
    "drake/examples/TRI_Remy/remy_description/robot/jaco_remy_spheres.urdf";
const char* kRelBlockUrdfPath =
    "drake/examples/TRI_Remy/remy_description/objects/block.urdf";

int DoMain() {
  drake::lcm::DrakeLcm lcm;
  systems::DiagramBuilder<double> builder;

  // Adds a plant.
  RigidBodyPlant<double> *plant = nullptr;
  RemyControllerSystem<double>* controller = nullptr;

  {
    auto tree = std::make_unique<RigidBodyTree<double>>();
    drake::multibody::AddFlatTerrainToWorld(tree.get());

    Eigen::Isometry3d pose(Eigen::Translation<double, 3>(0, 0,FLAGS_z0));
    CreateTreeFromFloatingModelAtPose(
        FindResourceOrThrow(kRelUrdfPath), tree.get(), pose);


    pose = Eigen::Translation<double, 3>(FLAGS_bx, FLAGS_by, FLAGS_bz);
    CreateTreeFromFloatingModelAtPose(FindResourceOrThrow(kRelBlockUrdfPath),
                                      tree.get(), pose);

    pose = Eigen::Translation<double, 3>(FLAGS_bx+0.1, FLAGS_by-.25, FLAGS_bz);
    CreateTreeFromFloatingModelAtPose(FindResourceOrThrow(kRelBlockUrdfPath),
                                      tree.get(), pose);

    const double dynamic_friction_coeff = 0.5;
    const double static_friction_coeff = 1.0;
    const double v_stiction_tolerance = 0.05;

    auto control_tree = tree->Clone();

    plant = builder.AddSystem<RigidBodyPlant<double>>(std::move(tree));
    plant->set_name("plant");
    plant->set_normal_contact_parameters(FLAGS_stiffness,
                                         FLAGS_dissipation);
    plant->set_friction_contact_parameters(
        static_friction_coeff, dynamic_friction_coeff, v_stiction_tolerance);

    controller = builder.AddSystem<RemyControllerSystem<double>>(
        std::move(control_tree));
  }

  // Verifies the tree.
  const RigidBodyTree<double> &tree = plant->get_rigid_body_tree();
  VerifyRemyTree(tree);
  //PrintOutRemyTree(tree);

  // Creates and adds LCM publisher for visualization.
  auto visualizer = builder.AddSystem<systems::DrakeVisualizer>(tree, &lcm);

  builder.Connect(controller->get_output_port_control(),
                  plant->model_instance_actuator_command_input_port(0));
  builder.Connect(plant->get_output_port(0),
                  controller->get_input_port_full_estimated_state());

  // Connects the visualizer and builds the diagram.
  builder.Connect(plant->get_output_port(0), visualizer->get_input_port(0));

  std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();
  systems::Simulator<double> simulator(*diagram);

  Context<double>& remy_context = diagram->GetMutableSubsystemContext(
      *plant, simulator.get_mutable_context());

  // Sets torso lift initial conditions.
  // See the @file docblock in remy_common.h for joint index descriptions.
  VectorBase<double> *x0 = remy_context.get_mutable_continuous_state_vector();
  const int kLiftJointIdx = 9;
  x0->SetAtIndex(kLiftJointIdx, 0.4);

  // set torso joint limit dynamics
  const int kJointStiff = 10000; // stiffness
  const int kJointDiss = 50; // dissipation
  const int kTorsoIndex = tree.FindBodyIndex("lift");
  PrismaticJoint& tjoint = (PrismaticJoint&)tree.get_body(kTorsoIndex).getJoint();
  tjoint.SetJointLimitDynamics(kJointStiff,kJointDiss);

  // initialize and run the simulation
  simulator.Initialize();

  // Simulate for the desired duration.
  simulator.set_target_realtime_rate(1);
  simulator.StepTo(7);

  return 0;
}

}  // namespace Remy
}  // namespace examples
}  // namespace drake

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::Remy::DoMain();
}
