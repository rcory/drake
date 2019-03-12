#include <iostream>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/common/is_approx_equal_abstol.h"
#include "drake/examples/pendulum/pendulum_plant.h"
#include "drake/examples/pendulum/trajectory_optimization_common.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/pid_controlled_system.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/systems/trajectory_optimization/direct_collocation.h"
#include "drake/systems/trajectory_optimization/direct_transcription.h"

using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::multibody::UniformGravityFieldElement;

namespace drake {
namespace examples {
namespace pendulum {

using trajectories::PiecewisePolynomial;

namespace {

DEFINE_double(target_realtime_rate, 1.0,
              "Playback speed.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_bool(use_dircol, true,
              "Indicates whether the optimization uses DirectCollocation. If"
              "this is false, use DirectTranscription.");

std::unique_ptr<MultibodyPlant<double>> BuildPlant(
    double time_step, geometry::SceneGraph<double>* scene_graph) {
  const char* const urdf_path = "drake/examples/pendulum/Pendulum.urdf";
  auto spendulum = std::make_unique<MultibodyPlant<double>>(time_step);
  spendulum->AddForceElement<UniformGravityFieldElement>();
  Parser sparser(spendulum.get(), scene_graph);
  sparser.AddModelFromFile(FindResourceOrThrow(urdf_path));
  spendulum->WeldFrames(spendulum->world_frame(),
                        spendulum->GetFrameByName("base_part2"));

  spendulum->Finalize(scene_graph);
  spendulum->set_name("spendulum");
  return spendulum;
}

int DoMain() {
  // Declare some common simulation components.
  systems::DiagramBuilder<double> builder;
  auto scene_graph = builder.AddSystem<geometry::SceneGraph>();

  if (FLAGS_use_dircol) {
    drake::log()->info("Using dircol");
    auto pendulum = BuildPlant(0, nullptr);
    auto context = pendulum->CreateDefaultContext();
    const int actuation_port_index =
        pendulum->get_actuation_input_port().get_index();

    const int kNumTimeSamples = 21;
    const double kMinimumTimeStep = 0.2;
    const double kMaximumTimeStep = 0.5;
    systems::trajectory_optimization::DirectCollocation dircol(
        pendulum.get(), *context, kNumTimeSamples, kMinimumTimeStep,
        kMaximumTimeStep, actuation_port_index);

    dircol.AddEqualTimeIntervalsConstraints();
    AddSwingupConstraints(&dircol);

    const auto result = solvers::Solve(dircol);
    if (!ResultIsSuccess(result)) {
      return 1;
    }
    // Simulate
    auto sim_pendulum = BuildPlant(0.0, scene_graph);  // Continuous time MBP.
    const geometry::SourceId source_id = sim_pendulum->get_source_id().value();
    SimulateTrajectory(scene_graph, source_id, std::move(sim_pendulum), &dircol,
                       result, &builder, FLAGS_target_realtime_rate);
  } else {
    drake::log()->info("Using dirtran");
    auto dpendulum = BuildPlant(0.05, nullptr);
    auto dcontext = dpendulum->CreateDefaultContext();
    const int actuation_port_index =
        dpendulum->get_actuation_input_port().get_index();

    const int kNumTimeSamples = 100;
    systems::trajectory_optimization::DirectTranscription dirtran(
        dpendulum.get(), *dcontext, kNumTimeSamples, actuation_port_index);

    AddSwingupConstraints(&dirtran);

    const auto result = solvers::Solve(dirtran);
    if (!ResultIsSuccess(result)) {
      return 1;
    }
    // Simulate
    auto sim_pendulum = BuildPlant(0.0, scene_graph);  // Continuous time MBP.
    const geometry::SourceId source_id = sim_pendulum->get_source_id().value();
    SimulateTrajectory(scene_graph, source_id, std::move(sim_pendulum),
                       &dirtran, result, &builder, FLAGS_target_realtime_rate);
  }
  return 0;
}

}  // namespace
}  // namespace pendulum
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::pendulum::DoMain();
}
