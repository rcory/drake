#include <iostream>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/common/is_approx_equal_abstol.h"
#include "drake/examples/pendulum/pendulum_plant.h"
#include "drake/examples/pendulum/trajectory_optimization_common.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/pid_controlled_system.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/systems/trajectory_optimization/direct_collocation.h"
#include "drake/systems/trajectory_optimization/direct_transcription.h"

using drake::solvers::SolutionResult;

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

int DoMain() {
  // Declare some common simulation components.
  systems::DiagramBuilder<double> builder;
  auto scene_graph = builder.AddSystem<geometry::SceneGraph>();

  if (FLAGS_use_dircol) {
    auto pendulum = std::make_unique<PendulumPlant<double>>();
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
    // Run the simulation.
    pendulum->RegisterGeometry(pendulum->get_parameters(*context), scene_graph);
    const geometry::SourceId source_id = pendulum->source_id();
    SimulateTrajectory(scene_graph, source_id, std::move(pendulum), &dircol,
                       result, &builder, FLAGS_target_realtime_rate);
  } else {
    // DirectTranscription uses a discrete plant.
    auto dpendulum = std::make_unique<PendulumPlant<double>>(0.05);
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
    // Run the simulation with the continuous time plant.
    auto pendulum = std::make_unique<PendulumPlant<double>>();
    auto context = pendulum->CreateDefaultContext();
    pendulum->RegisterGeometry(pendulum->get_parameters(*context), scene_graph);
    const geometry::SourceId source_id = pendulum->source_id();
    SimulateTrajectory(scene_graph, source_id, std::move(pendulum), &dirtran,
                       result, &builder, FLAGS_target_realtime_rate);
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
