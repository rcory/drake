#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/common/text_logging_gflags.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/plant/force_controller.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/zero_order_hold.h"

namespace drake {
namespace examples {
namespace planar_gripper {
namespace {

using geometry::SceneGraph;
using multibody::MultibodyPlant;
using multibody::Parser;
using lcm::DrakeLcm;
using multibody::ContactResults;
using systems::ZeroOrderHold;

DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_double(simulation_time, 8.0,
              "Desired duration of the simulation in seconds.");
DEFINE_double(time_step, 1e-4,
            "If greater than zero, the plant is modeled as a system with "
            "discrete updates and period equal to this time_step. "
            "If 0, the plant is modeled as a continuous system.");

 class ForceController : public systems::LeafSystem<double> {
  public:
   DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ForceController)
  
   ForceController() {
         this->DeclareAbstractInputPort(Value<ContactResults<double>>{});
         this->DeclareVectorOutputPort(
                 "tau", systems::BasicVector<double>(7),
                 &ForceController::CalcOutput);
   }
   void CalcOutput(const systems::Context<double>& context,
                   systems::BasicVector<double>* output_vector) const {
     auto torque_calc = output_vector->get_mutable_value();

     const auto& contact_results =
         this->get_input_port(0).Eval<ContactResults<double>>(context);
     unused(contact_results);
//      unused(context);

     torque_calc.setZero();
   }
 };

int do_main() {
  systems::DiagramBuilder<double> builder;

  SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
  scene_graph.set_name("scene_graph");

  const std::string full_name = FindResourceOrThrow(
      "drake/manipulation/models/iiwa_description/urdf/"
      "iiwa14_spheres_collision.urdf");

  MultibodyPlant<double>& plant =
      *builder.AddSystem<MultibodyPlant>(FLAGS_time_step);
  Parser(&plant, &scene_graph).AddModelFromFile(full_name);
  const multibody::Frame<double> &base_frame =
      plant.GetFrameByName("base");
  plant.WeldFrames(plant.world_frame(), base_frame);

  // Now the model is complete.
  plant.Finalize();

  // Sanity check on the availability of the optional source id before using it.
  DRAKE_DEMAND(plant.geometry_source_is_registered());


  DrakeLcm lcm;
  geometry::ConnectDrakeVisualizer(&builder, scene_graph, &lcm);

//  //    Publish contact results for visualization.
//  drake::multibody::ConnectContactResultsToDrakeVisualizer(&builder, plant,
//                                                           &lcm);

   // Connect the force controller.
  auto zoh = builder.AddSystem<systems::ZeroOrderHold<double>>(
      1e-3, Value<ContactResults<double>>());
  auto force_controller = builder.AddSystem<ForceController>();
  builder.Connect(plant.get_contact_results_output_port(),
                  zoh->get_input_port());
  builder.Connect(zoh->get_output_port(), force_controller->get_input_port(0));
  builder.Connect(force_controller->get_output_port(0),
                  plant.get_actuation_input_port());

  //  // Connect a const vector source instead.
  //   Eigen::VectorXd const_vec(7);
  //   const_vec.setZero();
  //   auto const_src =
  //       builder.AddSystem<systems::ConstantVectorSource<double>>(const_vec);
  //   builder.Connect(const_src->get_output_port(),
  //                   plant.get_actuation_input_port());

  // Connect MBP snd SG.
  builder.Connect(
      plant.get_geometry_poses_output_port(),
      scene_graph.get_source_pose_port(plant.get_source_id().value()));
  builder.Connect(scene_graph.get_query_output_port(),
                  plant.get_geometry_query_input_port());

  auto diagram = builder.Build();

  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
//   diagram_context->DisableCaching();
  diagram->SetDefaultContext(diagram_context.get());

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

   simulator.set_publish_every_time_step(false);
   simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
   simulator.Initialize();
   simulator.AdvanceTo(FLAGS_simulation_time);

   return 0;
}

}  // namespace
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "A simple planar gripper example.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::logging::HandleSpdlogGflags();
  return drake::examples::planar_gripper::do_main();
}
