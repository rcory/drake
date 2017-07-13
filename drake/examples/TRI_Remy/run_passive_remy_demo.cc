/// @file
///
/// This test sets up a simple passive dynamics simulation of the  mobile
/// robot, i.e., all joint torques are set to zero.

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/common/text_logging_gflags.h"
#include "drake/examples/TRI_Remy/remy_common.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/rigid_body_plant/drake_visualizer.h"
#include "drake/multibody/rigid_body_plant/rigid_body_plant.h"
#include "drake/multibody/rigid_body_tree_construction.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_vector_source.h"

namespace drake {

using systems::Context;
using systems::ContinuousState;
using systems::RigidBodyPlant;
using systems::VectorBase;

namespace examples {
namespace Remy {

int DoMain() {
  drake::lcm::DrakeLcm lcm;
  systems::DiagramBuilder<double> builder;

  // Adds a plant.
  RigidBodyPlant<double> *plant = nullptr;
  {
    auto tree = std::make_unique<RigidBodyTree<double>>();
    drake::multibody::AddFlatTerrainToWorld(tree.get());

    Eigen::Isometry3d pose(Eigen::Translation<double, 3>(0, 0, 0.11));
    CreateTreeFromFloatingModelAtPose(
        FindResourceOrThrow(
            "drake/examples/TRI_Remy/remy_description/robot/remy_clumsy.urdf"),
        tree.get(), pose);

    const double contact_stiffness = 50000;
    const double contact_dissipation = 2;
    const double dynamic_friction_coeff = 0.5;
    const double static_friction_coeff = 1.0;
    const double v_stiction_tolerance = 0.05;

    plant = builder.AddSystem<RigidBodyPlant<double>>(std::move(tree));
    plant->set_name("plant");
    plant->set_normal_contact_parameters(contact_stiffness,
                                         contact_dissipation);
    plant->set_friction_contact_parameters(
        static_friction_coeff, dynamic_friction_coeff, v_stiction_tolerance);
  }

  // Verifies the tree.
  const RigidBodyTree<double> &tree = plant->get_rigid_body_tree();
  // VerifyRemyTree(tree);

  // Creates and adds LCM publisher for visualization.
  auto visualizer = builder.AddSystem<systems::DrakeVisualizer>(tree, &lcm);

  Eigen::Matrix<double, 11, 1> input_values;
  input_values << 1, 1, 0, 0, 0, 0, 0, 0, 0, 0;

  auto zero_source =
      builder.AddSystem<systems::ConstantVectorSource<double>>(input_values);
  zero_source->set_name("zero_source");
  builder.Connect(zero_source->get_output_port(), plant->get_input_port(0));

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
  const int kShoulderForeAftIdx = 15;
  const int kElbowForeAftIdx = 16;
  x0->SetAtIndex(kLiftJointIdx, 0.4);

  x0->SetAtIndex(14, -1.57);
  x0->SetAtIndex(kShoulderForeAftIdx, -1.57);
  x0->SetAtIndex(kElbowForeAftIdx, 0);

  simulator.Initialize();

  // Simulate for the desired duration.
  simulator.set_target_realtime_rate(1);
  simulator.StepTo(7);

  //  // Ensures the simulation was successful.
  //  const Context<double> &context = simulator.get_context();
  //  const ContinuousState<double> *state = context.get_continuous_state();
  //  const VectorBase<double> &position_vector =
  //  state->get_generalized_position();
  //  const VectorBase<double> &velocity_vector =
  //  state->get_generalized_velocity();

  return 0;
}

}  // namespace Remy
}  // namespace examples
}  // namespace drake

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::Remy::DoMain();
}
