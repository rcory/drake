#include "drake/common/find_resource.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_common.h"
#include "drake/lcmtypes/drake/lcmt_contact_results_for_viz.hpp"
#include "drake/multibody/rigid_body_plant/contact_results_to_lcm.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_world/world_sim_tree_builder.h"
#include "drake/examples/kuka_iiwa_arm/sim_diagram_builder.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/rigid_body_tree_construction.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/primitives/trajectory_source.h"


namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace {

std::unique_ptr<RigidBodyTree<double>> build_tree(
    std::vector<ModelInstanceInfo<double>>* iiwa,
    ModelInstanceInfo<double>* box) {
  auto tree_builder = std::make_unique<WorldSimTreeBuilder<double>>();

  // Adds models to the simulation builder. Instances of these models can be
  // subsequently added to the world.
  tree_builder->StoreModel(
      "iiwa_l",
      "drake/manipulation/models/iiwa_description/urdf/iiwa14_l.urdf");

  tree_builder->StoreModel(
      "iiwa_r",
      "drake/manipulation/models/iiwa_description/urdf/iiwa14_r.urdf");

  tree_builder->StoreModel(
      "box",
      "drake/examples/kuka_iiwa_arm/dev/box_rotation/box.urdf");

  iiwa->clear();

  // Left arm.
  int id = tree_builder->AddFixedModelInstance(
      "iiwa_l", Vector3<double>(0, 0.96, 0));
  iiwa->push_back(tree_builder->get_model_info_for_instance(id));

  // Right arm.
  id = tree_builder->AddFixedModelInstance(
      "iiwa_r", Vector3<double>(0, 0, 0));
  iiwa->push_back(tree_builder->get_model_info_for_instance(id));

  // Box location, box sides is 0.2 0.2 0.2
  id = tree_builder->AddFloatingModelInstance("box",
      Vector3<double>(0.5, 0.5, 0.28));
  *box = tree_builder->get_model_info_for_instance(id);

  auto tree = tree_builder->Build();
  multibody::AddFlatTerrainToWorld(tree.get(), 100., 10.);

  /*
  // Make a stand for the box.
  Isometry3<double> stand(Isometry3<double>::Identity());
  stand.translation() << 0.6, -0.4, -0.1;
  const Vector3<double> stand_sides(0.3, 0.3, 0.3);
  multibody::AddBoxToWorld(stand, stand_sides, "stand", tree.get());
  */

  return tree;
}

MatrixX<double> get_posture(const std::string& name) {
  std::fstream fs;
  fs.open(name, std::fstream::in);
  DRAKE_DEMAND(fs.is_open());

  MatrixX<double> ret(12, 21);
  for (int i = 0; i < ret.rows(); ++i) {
    for (int j = 0; j < ret.cols(); ++j) {
      fs >> ret(i, j);
    }
  }
  return ret;
}

void main() {

  MatrixX<double> keyframes = get_posture(
      "drake/examples/kuka_iiwa_arm/dev/box_rotation/simple_keyframes.txt");

  drake::lcm::DrakeLcm lcm;
  std::vector<ModelInstanceInfo<double>> iiwa_info;
  ModelInstanceInfo<double> box_info;
  SimDiagramBuilder<double> builder;
  systems::DiagramBuilder<double>* diagram_builder =
      builder.get_mutable_builder();

  auto plant = builder.AddPlant(build_tree(&iiwa_info, &box_info));

  builder.AddVisualizer(&lcm);

  std::vector<systems::TrajectorySource<double>*> iiwa_traj_src(iiwa_info.size());

  const int N = keyframes.rows();
  std::vector<double> times(N);
  for (int i = 0; i < N; ++i) {
    if (i == 0)
      times[i] = keyframes(i, 0);
    else
      times[i] = times[i - 1] + keyframes(i, 0);
  }

  // Left arm traj.
  {
    std::vector<MatrixX<double>> knots(N, MatrixX<double>::Zero(7, 1));
    std::vector<MatrixX<double>> knotsd(N, MatrixX<double>::Zero(7, 1));

    for (int i = 0; i < N; ++i)
      knots[i] = keyframes.block<1, 7>(i, 1).transpose();

    //PiecewisePolynomial<double> poly = PiecewisePolynomial<double>::FirstOrderHold(
    //    times, knots);
    PiecewisePolynomial<double> poly = PiecewisePolynomial<double>::Cubic(
        times, knots, knotsd);

    // Adds a trajectory source for desired state and accelerations.
    iiwa_traj_src[0] =
        diagram_builder->template AddSystem<systems::TrajectorySource<double>>(
            PiecewisePolynomialTrajectory(poly), 1);
  }

  // Right arm traj.
  {
    std::vector<MatrixX<double>> knots(N, MatrixX<double>::Zero(7, 1));
    std::vector<MatrixX<double>> knotsd(N, MatrixX<double>::Zero(7, 1));

    for (int i = 0; i < N; ++i)
      knots[i] = keyframes.block<1, 7>(i, 8).transpose();

    //PiecewisePolynomial<double> poly = PiecewisePolynomial<double>::FirstOrderHold(
    //    times, knots);
    PiecewisePolynomial<double> poly = PiecewisePolynomial<double>::Cubic(
        times, knots, knotsd);

    // Adds a trajectory source for desired state and accelerations.
    iiwa_traj_src[1] =
        diagram_builder->template AddSystem<systems::TrajectorySource<double>>(
            PiecewisePolynomialTrajectory(poly), 1);
  }

  // Adds controllers for all the iiwa arms.
  VectorX<double> iiwa_kp, iiwa_kd, iiwa_ki;
  SetPositionControlledIiwaGains(&iiwa_kp, &iiwa_ki, &iiwa_kd);
  int ctr = 0;
  for (const auto& info : iiwa_info) {
    auto controller =
        builder.AddController<systems::controllers::InverseDynamicsController<double>>(
            info.instance_id, plant->get_rigid_body_tree().Clone(), iiwa_kp,
            iiwa_ki, iiwa_kd, false /* no feedforward acceleration */);

    controller->set_name("controller" + std::to_string(ctr));

    diagram_builder->Connect(iiwa_traj_src[ctr]->get_output_port(),
                             controller->get_input_port_desired_state());
    ctr++;
  }

  // Add contact viz.
  auto contact_viz =
      diagram_builder->AddSystem<systems::ContactResultsToLcmSystem<double>>(
          plant->get_rigid_body_tree());
  contact_viz->set_name("contact_viz");

  auto contact_results_publisher = diagram_builder->AddSystem(
      systems::lcm::LcmPublisherSystem::Make<lcmt_contact_results_for_viz>(
          "CONTACT_RESULTS", &lcm));
  contact_results_publisher->set_name("contact_results_publisher");

  diagram_builder->Connect(plant->contact_results_output_port(),
      contact_viz->get_input_port(0));
  diagram_builder->Connect(contact_viz->get_output_port(0),
      contact_results_publisher->get_input_port(0));

  // Simulates.
  std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();
  systems::Simulator<double> simulator(*diagram);

  // Arbitrary contact parameters.
  const double kStiffness = 10000;
  const double kDissipation = 2.0;
  const double kStaticFriction = 10;
  const double kDynamicFriction = 1;
  const double kVStictionTolerance = 0.01;
  plant->set_normal_contact_parameters(kStiffness, kDissipation);
  plant->set_friction_contact_parameters(kStaticFriction, kDynamicFriction,
                                         kVStictionTolerance);

  simulator.get_mutable_integrator()->set_maximum_step_size(1e-3);

  // simulator.reset_integrator<systems::RungeKutta2Integrator<double>>(*diagram, simulator.get_mutable_context());
  // simulator.get_mutable_integrator()->request_initial_step_size_target(3e-4);
  // simulator.get_mutable_integrator()->set_target_accuracy(1e-3);

  simulator.Initialize();
  simulator.set_target_realtime_rate(1.0);

  int next_keyframe = 0;
  while (true) {
    double t = simulator.get_context().get_time();
    simulator.StepTo(t + 0.01);
    if (next_keyframe < N && t >= times[next_keyframe]) {
      std::cout << "exec knot: " << next_keyframe << "\n";
      next_keyframe++;
    }
  }
}

}  // namespace
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main() {
  drake::examples::kuka_iiwa_arm::main();

  return 0;
}
