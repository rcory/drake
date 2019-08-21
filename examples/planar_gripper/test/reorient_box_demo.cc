#include <limits>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/common/proto/call_python.h"
#include "drake/common/text_logging_gflags.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/examples/planar_gripper/gripper_brick_trajectory_optimization.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/signal_logger.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/systems/rendering/multibody_position_to_geometry_pose.h"

namespace drake {
namespace examples {
namespace planar_gripper {

const double kInf = std::numeric_limits<double>::infinity();

DEFINE_double(target_realtime_rate, 1,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_double(simulation_time, 3,
              "Desired duration of the simulation in seconds.");
DEFINE_double(time_step, 1e-4,
              "If greater than zero, the plant is modeled as a system with "
              "discrete updates and period equal to this time_step. "
              "If 0, the plant is modeled as a continuous system.");
DEFINE_string(visualization, "plan",
              "visualization option could be 'plan', 'simulation' or 'both'");

void VisualizeTrajectory(const trajectories::Trajectory<double>& traj) {
  systems::DiagramBuilder<double> builder;
  auto plant = std::make_unique<multibody::MultibodyPlant<double>>();
  auto scene_graph = builder.AddSystem<geometry::SceneGraph<double>>();
  plant->RegisterAsSourceForSceneGraph(scene_graph);
  const std::string gripper_path =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_gripper.sdf");
  multibody::Parser parser(plant.get(), scene_graph);
  parser.AddModelFromFile(gripper_path, "gripper");
  examples::planar_gripper::WeldGripperFrames(plant.get());
  const std::string brick_path =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_brick.sdf");
  parser.AddModelFromFile(brick_path, "brick");
  (plant)->WeldFrames((plant)->world_frame(),
                      (plant)->GetFrameByName("brick_base"),
                      math::RigidTransformd());

  (plant)->Finalize();

  auto traj_source =
      builder.AddSystem<systems::TrajectorySource<double>>(traj, 0, true);
  auto to_pose =
      builder.AddSystem<systems::rendering::MultibodyPositionToGeometryPose>(
          *plant);
  builder.Connect(traj_source->get_output_port(), to_pose->get_input_port());
  builder.Connect(
      to_pose->get_output_port(),
      scene_graph->get_source_pose_port(plant->get_source_id().value()));

  ConnectDrakeVisualizer(&builder, *scene_graph);

  auto diagram = builder.Build();
  systems::Simulator<double> simulator(*diagram);
  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.AdvanceTo(traj.end_time());
}

std::unordered_map<Finger, Eigen::VectorXd> ComputeFingerBrickDistance(
    const GripperBrickHelper<double>& gripper_brick, const Eigen::MatrixXd& q) {
  auto diagram_context_sample = gripper_brick.diagram().CreateDefaultContext();
  auto plant_context_sample =
      &(gripper_brick.diagram().GetMutableSubsystemContext(
          gripper_brick.plant(), diagram_context_sample.get()));
  std::unordered_map<Finger, Eigen::VectorXd> finger_face_distance;
  finger_face_distance.emplace(Finger::kFinger1,
                               Eigen::VectorXd::Zero(q.cols()));
  finger_face_distance.emplace(Finger::kFinger2,
                               Eigen::VectorXd::Zero(q.cols()));
  finger_face_distance.emplace(Finger::kFinger3,
                               Eigen::VectorXd::Zero(q.cols()));
  for (int i = 0; i < q.cols(); ++i) {
    gripper_brick.plant().SetPositions(plant_context_sample, q.col(i));
    const auto& query_port =
        gripper_brick.plant().get_geometry_query_input_port();
    const auto& query_object =
        query_port.Eval<geometry::QueryObject<double>>(*plant_context_sample);
    const std::vector<geometry::SignedDistancePair<double>>
        signed_distance_pairs =
            query_object.ComputeSignedDistancePairwiseClosestPoints(kInf);
    for (const auto& signed_distance_pair : signed_distance_pairs) {
      for (Finger finger :
           {Finger::kFinger1, Finger::kFinger2, Finger::kFinger3}) {
        if (SortedPair<geometry::GeometryId>(signed_distance_pair.id_A,
                                             signed_distance_pair.id_B) ==
            SortedPair<geometry::GeometryId>(
                gripper_brick.finger_tip_sphere_geometry_id(finger),
                gripper_brick.brick_geometry_id())) {
          finger_face_distance[finger](i) = signed_distance_pair.distance;
        }
      }
    }
  }
  return finger_face_distance;
}

void PlotFingerFaceDistance(const GripperBrickHelper<double>& gripper_brick,
                            const Eigen::VectorXd& t_sample,
                            const Eigen::MatrixXd& q_sample) {
  const std::unordered_map<Finger, Eigen::VectorXd> finger_face_distance =
      ComputeFingerBrickDistance(gripper_brick, q_sample);
  common::CallPython("print", "plot finger face distance");
  common::CallPython("figure");
  common::CallPython("title", "Finger/brick distance");
  common::CallPython("plot", t_sample,
                     finger_face_distance.at(Finger::kFinger1),
                     common::ToPythonKwargs("label", "finger 1"));
  common::CallPython("plot", t_sample,
                     finger_face_distance.at(Finger::kFinger2),
                     common::ToPythonKwargs("label", "finger 2"));
  common::CallPython("plot", t_sample,
                     finger_face_distance.at(Finger::kFinger3),
                     common::ToPythonKwargs("label", "finger 3"));
  common::CallPython("legend");
  common::CallPython("xlabel", "time (s)");
  common::CallPython("ylabel", "distance (m)");
  common::CallPython("show");
}

std::tuple<trajectories::PiecewisePolynomial<double>, Eigen::Vector3d,
           std::map<std::string, int>,
           trajectories::PiecewisePolynomial<double>>
GenerateReorientationTrajectory() {
  GripperBrickHelper<double> gripper_brick;
  int nT = 19;
  std::map<Finger, BrickFace> initial_contact(
      {{Finger::kFinger1, BrickFace::kNegY},
       {Finger::kFinger2, BrickFace::kNegZ},
       {Finger::kFinger3, BrickFace::kPosY}});

  std::vector<FingerTransition> finger_transitions;
  finger_transitions.emplace_back(2, 5, Finger::kFinger1, BrickFace::kPosZ);
  finger_transitions.emplace_back(8, 11, Finger::kFinger3, BrickFace::kNegZ);
  finger_transitions.emplace_back(14, 17, Finger::kFinger2, BrickFace::kNegY);

  const double brick_lid_friction_force_magnitude = 0;
  const double brick_lid_friction_torque_magnitude = 0;

  const double depth = 0e-4;
  const double friction_cone_shrink_factor = 0.9;
  GripperBrickTrajectoryOptimization dut(
      &gripper_brick, nT, initial_contact, finger_transitions,
      brick_lid_friction_force_magnitude, brick_lid_friction_torque_magnitude,
      GripperBrickTrajectoryOptimization::Options(
          0.8, GripperBrickTrajectoryOptimization::IntegrationMethod::kMidpoint,
          0.03 * M_PI, 0.03, depth, friction_cone_shrink_factor));

  dut.get_mutable_prog()->AddBoundingBoxConstraint(0.1, 0.15, dut.dt());

  // Initial pose constraint on the brick.
  dut.get_mutable_prog()->AddBoundingBoxConstraint(
      0, 0, dut.q_vars()(gripper_brick.brick_revolute_x_position_index(), 0));

  // Final pose constraint on the brick.
  dut.get_mutable_prog()->AddBoundingBoxConstraint(
      0.3 * M_PI, 0.4 * M_PI,
      dut.q_vars()(gripper_brick.brick_revolute_x_position_index(), nT - 1));

  // Fingers cannot move too fast
  for (const auto& finger_transition : finger_transitions) {
    for (int i = finger_transition.start_knot_index;
         i < finger_transition.end_knot_index; ++i) {
      dut.AddPositionDifferenceBound(
          i, gripper_brick.finger_base_position_index(finger_transition.finger),
          0.1 * M_PI);
      dut.AddPositionDifferenceBound(
          i, gripper_brick.finger_mid_position_index(finger_transition.finger),
          0.1 * M_PI);
    }
  }

  dut.AddBrickStaticEquilibriumConstraint(0);
  dut.AddBrickStaticEquilibriumConstraint(nT - 1);
  dut.get_mutable_prog()->SetSolverOption(solvers::SnoptSolver::id(),
                                          "iterations limit", 100000);

  solvers::SnoptSolver solver;
  const solvers::MathematicalProgramResult result =
      solvers::Solve(dut.prog(), {}, {});
  // std::cout << "info: "
  //          << result.get_solver_details<solvers::SnoptSolver>().info << "\n";
  std::cout << result.get_solution_result() << "\n";
  if (!result.is_success()) {
    for (const auto& constraint : dut.prog().GetAllConstraints()) {
      const Eigen::VectorXd constraint_val = result.EvalBinding(constraint);
      if (((constraint_val.array() -
            constraint.evaluator()->lower_bound().array()) < -1E-5)
              .any() ||
          ((constraint_val.array() -
            constraint.evaluator()->upper_bound().array()) > 1E-5)
              .any()) {
        std::cout << constraint.evaluator()->get_description() << "\n";
        std::cout << constraint.variables().transpose() << "\n";
      }
    }
  }

  const Eigen::MatrixXd q_sol = result.GetSolution(dut.q_vars());
  const Eigen::VectorXd dt_sol = result.GetSolution(dut.dt());
  std::cout << "dt: " << dt_sol.transpose() << "\n";
  std::cout << "total time: " << dt_sol.sum() << "\n";

  // auto diagram_context = gripper_brick.diagram().CreateDefaultContext();
  // systems::Context<double>* plant_mutable_context =
  //    &(gripper_brick.diagram().GetMutableSubsystemContext(
  //        gripper_brick.plant(), diagram_context.get()));
  // for (int i = 0; i < nT; ++i) {
  //  gripper_brick.plant().SetPositions(plant_mutable_context, q_sol.col(i));
  //  gripper_brick.diagram().Publish(*diagram_context);
  //  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  //}
  Eigen::Vector3d brick_initial_pose(
      q_sol(gripper_brick.brick_translate_y_position_index(), 0),
      q_sol(gripper_brick.brick_translate_z_position_index(), 0),
      q_sol(gripper_brick.brick_revolute_x_position_index(), 0));

  const std::map<std::string, int> finger_joint_name_to_row_index_map(
      {{"finger1_BaseJoint", 0},
       {"finger2_BaseJoint", 1},
       {"finger3_BaseJoint", 2},
       {"finger1_MidJoint", 3},
       {"finger2_MidJoint", 4},
       {"finger3_MidJoint", 5}});

  const trajectories::PiecewisePolynomial<double> finger_traj =
      dut.ReconstructFingerTrajectory(result,
                                      finger_joint_name_to_row_index_map);

  const Eigen::VectorXd t_sol = dut.ReconstructTimeSolution(result);
  const trajectories::PiecewisePolynomial<double> q_traj =
      trajectories::PiecewisePolynomial<double>::FirstOrderHold(t_sol, q_sol);

  const trajectories::PiecewisePolynomial<double> finger1_force_traj =
      dut.ReconstructFingerForceTrajectory(Finger::kFinger1, result);

  const trajectories::PiecewisePolynomial<double> q_refined_traj =
      dut.RefineTrajectory(result);
  if (FLAGS_visualization == "plan" || FLAGS_visualization == "both") {
    VisualizeTrajectory(q_refined_traj);
  }
  const Eigen::VectorXd t_sample =
      Eigen::VectorXd::LinSpaced(2000, 0, q_refined_traj.end_time());
  Eigen::MatrixXd q_sample(q_sol.rows(), t_sample.rows());
  for (int i = 0; i < t_sample.rows(); ++i) {
    q_sample.col(i) = q_refined_traj.value(t_sample(i));
  }

  PlotFingerFaceDistance(gripper_brick, t_sample, q_sample);

  trajectories::PiecewisePolynomial<double> finger_refined_traj =
      dut.ReconstructFingerTrajectory(q_refined_traj,
                                      finger_joint_name_to_row_index_map);
  std::cout << "planned x0: " << q_refined_traj.value(0).transpose() << "\n";

  return std::make_tuple(finger_refined_traj, brick_initial_pose,
                         finger_joint_name_to_row_index_map, q_refined_traj);
}  // namespace planar_gripper

namespace {

using geometry::SceneGraph;
using math::RigidTransform;
using math::RollPitchYaw;
using multibody::JointActuatorIndex;
using multibody::ModelInstanceIndex;
using multibody::MultibodyPlant;
using multibody::Parser;
using multibody::PrismaticJoint;
using multibody::RevoluteJoint;

/// Converts the generalized force output of the ID controller (internally using
/// a control plant with only the gripper) to the generalized force input for
/// the full simulation plant (containing gripper and object).
class ControlToSimPlantForceConverter : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ControlToSimPlantForceConverter);
  ControlToSimPlantForceConverter(const MultibodyPlant<double>* plant,
                                  ModelInstanceIndex gripper_instance)
      : plant_(plant), gripper_instance_(gripper_instance) {
    DRAKE_DEMAND(plant != nullptr);
    this->DeclareVectorInputPort(
        "input", systems::BasicVector<double>(plant->num_actuators()));
    this->DeclareVectorOutputPort(
        "output", systems::BasicVector<double>(plant->num_velocities()),
        &ControlToSimPlantForceConverter::remap_output);
  }

  void remap_output(const systems::Context<double>& context,
                    systems::BasicVector<double>* output_vector) const {
    auto output_value = output_vector->get_mutable_value();
    auto input_value = this->EvalVectorInput(context, 0)->get_value();

    output_value.setZero();
    plant_->SetVelocitiesInArray(gripper_instance_, input_value, &output_value);
  }

 private:
  const MultibodyPlant<double>* plant_;
  const ModelInstanceIndex gripper_instance_;
};

std::tuple<Eigen::VectorXd, Eigen::MatrixXd> Simulate(
    const trajectories::PiecewisePolynomial<double>& finger_trajectory,
    const Eigen::Vector3d& brick_initial_pose,
    const std::map<std::string, int>& finger_joint_name_to_row_index_map) {
  systems::DiagramBuilder<double> builder;

  SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
  scene_graph.set_name("scene_graph");

  // Make and add the planar_gripper model.
  const std::string full_name =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_gripper.sdf");
  MultibodyPlant<double>& plant =
      *builder.AddSystem<MultibodyPlant>(FLAGS_time_step);
  const ModelInstanceIndex gripper_index =
      Parser(&plant, &scene_graph).AddModelFromFile(full_name);
  WeldGripperFrames<double>(&plant);

  // Adds the object to be manipulated.
  const std::string brick_file_name =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_brick.sdf");
  const ModelInstanceIndex brick_index =
      Parser(&plant).AddModelFromFile(brick_file_name, "object");
  const multibody::Frame<double>& brick_base_frame =
      plant.GetFrameByName("brick_base", brick_index);
  plant.WeldFrames(plant.world_frame(), brick_base_frame);

  // Create the controlled plant. Contains only the fingers (no objects).
  MultibodyPlant<double> control_plant(FLAGS_time_step);
  Parser(&control_plant).AddModelFromFile(full_name);
  WeldGripperFrames<double>(&control_plant);

  plant.Finalize();
  // plant.set_penetration_allowance(1e-3);
  control_plant.Finalize();

  // Sanity check on the availability of the optional source id before using it.
  DRAKE_DEMAND(plant.geometry_source_is_registered());

  // Create the gains for the inverse dynamics controller. These gains were
  // chosen arbitrarily.
  const int kNumJoints = 6;
  Vector<double, kNumJoints> Kp, Kd, Ki;
  Kp.setConstant(1500);
  Kd.setConstant(500);
  Ki.setConstant(500);

  auto id_controller =
      builder.AddSystem<systems::controllers::InverseDynamicsController>(
          control_plant, Kp, Ki, Kd, false);

  // Connect the ID controller.
  builder.Connect(plant.get_state_output_port(gripper_index),
                  id_controller->get_input_port_estimated_state());

  auto traj_src = builder.AddSystem<systems::TrajectorySource<double>>(
      finger_trajectory, 1 /* with one derivative */);
  builder.Connect(traj_src->get_output_port(),
                  id_controller->get_input_port_desired_state());

  // The inverse dynamics controller internally uses a "controlled plant", which
  // contains the gripper model *only* (i.e., no object). Therefore, its output
  // must be re-mapped to the input of the full "simulation plant", which
  // contains both gripper and object. The system
  // ControlToSimPlantForceConverter fills this role.
  auto generalized_force_map =
      builder.AddSystem<ControlToSimPlantForceConverter>(&plant, gripper_index);
  builder.Connect(*id_controller, *generalized_force_map);
  builder.Connect(generalized_force_map->get_output_port(0),
                  plant.get_applied_generalized_force_input_port());

  // Connect a constant zero vector to the actuation input port of MBP since we
  // don't use it (we use the generalized forces input).
  auto const_src = builder.AddSystem<systems::ConstantVectorSource>(
      VectorX<double>::Zero(kNumJoints));
  builder.Connect(const_src->get_output_port(),
                  plant.get_actuation_input_port());

  // Connect MBP snd SG.
  builder.Connect(
      plant.get_geometry_poses_output_port(),
      scene_graph.get_source_pose_port(plant.get_source_id().value()));
  builder.Connect(scene_graph.get_query_output_port(),
                  plant.get_geometry_query_input_port());

  auto signal_logger = builder.AddSystem<systems::SignalLogger<double>>(
      plant.num_positions() + plant.num_velocities(), 10000);
  builder.Connect(plant.get_state_output_port(),
                  signal_logger->get_input_port());

  geometry::ConnectDrakeVisualizer(&builder, scene_graph);
  lcm::DrakeLcm lcm;
  multibody::ConnectContactResultsToDrakeVisualizer(&builder, plant, &lcm);
  auto diagram = builder.Build();

  // Create a context for this system:
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  // Create the initial condition vector. Set initial joint velocities to zero.
  VectorX<double> gripper_initial_conditions =
      VectorX<double>::Zero(kNumJoints * 2);
  gripper_initial_conditions.head(kNumJoints) =
      finger_trajectory.value(0).block(0, 0, kNumJoints, 1);

  // All fingers consist of two joints: a base joint and a mid joint.
  // Set the initial finger joint positions.
  const int kNumFingers = 3;
  for (int i = 0; i < kNumFingers; i++) {
    std::string finger = "finger" + std::to_string(i + 1);
    const RevoluteJoint<double>& base_pin =
        plant.GetJointByName<RevoluteJoint>(finger + "_BaseJoint");
    const RevoluteJoint<double>& mid_pin =
        plant.GetJointByName<RevoluteJoint>(finger + "_MidJoint");
    int base_index =
        finger_joint_name_to_row_index_map.at(finger + "_BaseJoint");
    int mid_index = finger_joint_name_to_row_index_map.at(finger + "_MidJoint");
    base_pin.set_angle(&plant_context, gripper_initial_conditions(base_index));
    mid_pin.set_angle(&plant_context, gripper_initial_conditions(mid_index));
  }

  // Set the box initial conditions.
  const PrismaticJoint<double>& y_translate =
      plant.GetJointByName<PrismaticJoint>("brick_translate_y_joint");
  const PrismaticJoint<double>& z_translate =
      plant.GetJointByName<PrismaticJoint>("brick_translate_z_joint");
  const RevoluteJoint<double>& x_revolute =
      plant.GetJointByName<RevoluteJoint>("brick_revolute_x_joint");
  y_translate.set_translation(&plant_context, brick_initial_pose(0));
  z_translate.set_translation(&plant_context, brick_initial_pose(1));
  x_revolute.set_angle(&plant_context, brick_initial_pose(2));

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);

  std::cout << "signal logger x0: " << signal_logger->data().col(0).transpose()
            << "\n";
  const Eigen::VectorXd t_logged = signal_logger->sample_times();
  const Eigen::MatrixXd x_logged = signal_logger->data();
  return std::make_tuple(t_logged, x_logged);
}

}  // namespace
int DoMain() {
  trajectories::PiecewisePolynomial<double> finger_trajectory;
  Eigen::Vector3d brick_initial_pose;
  std::map<std::string, int> finger_joint_name_to_row_index_map;
  trajectories::PiecewisePolynomial<double> q_refined_traj;
  std::tie(finger_trajectory, brick_initial_pose,
           finger_joint_name_to_row_index_map, q_refined_traj) =
      GenerateReorientationTrajectory();
  if (FLAGS_visualization == "sim" || FLAGS_visualization == "both") {
    Eigen::VectorXd t_logged;
    Eigen::MatrixXd x_logged;
    std::tie(t_logged, x_logged) =
        Simulate(finger_trajectory, brick_initial_pose,
                 finger_joint_name_to_row_index_map);
    GripperBrickHelper<double> gripper_brick;
    Eigen::MatrixXd q_planned(gripper_brick.plant().num_positions(),
                              t_logged.rows());
    for (int i = 0; i < t_logged.rows(); ++i) {
      q_planned.col(i) = q_refined_traj.value(t_logged(i));
    }
    const Eigen::MatrixXd q_logged =
        x_logged.topRows(gripper_brick.plant().num_positions());

    PlotFingerFaceDistance(gripper_brick, t_logged, q_logged);

    common::CallPython("figure");
    int subplot_count = 1;
    for (Finger finger :
         {Finger::kFinger1, Finger::kFinger2, Finger::kFinger3}) {
      auto ax_l = common::CallPython("subplot", 3, 2, subplot_count++);
      common::CallPython(
          "plot", t_logged,
          x_logged.row(gripper_brick.finger_base_position_index(finger))
              .transpose());
      common::CallPython(
          "plot", t_logged,
          q_planned.row(gripper_brick.finger_base_position_index(finger))
              .transpose());
      ax_l.attr("set_title")(to_string(finger) + " base joint position");
      auto ax_r = common::CallPython("subplot", 3, 2, subplot_count++);
      common::CallPython(
          "plot", t_logged,
          x_logged.row(gripper_brick.finger_mid_position_index(finger))
              .transpose());
      common::CallPython(
          "plot", t_logged,
          q_planned.row(gripper_brick.finger_mid_position_index(finger))
              .transpose());
      ax_r.attr("set_title")(to_string(finger) + " middle joint position");
    }
    common::CallPython("show");

    common::CallPython("figure");
    subplot_count = 1;
    for (int i : {gripper_brick.brick_revolute_x_position_index(),
                  gripper_brick.brick_translate_y_position_index(),
                  gripper_brick.brick_translate_z_position_index()}) {
      auto ax = common::CallPython("subplot", 3, 1, subplot_count);
      common::CallPython("plot", t_logged, q_planned.row(i).transpose(),
                         common::ToPythonKwargs("label", "plan"));
      common::CallPython("plot", t_logged, x_logged.row(i).transpose(),
                         common::ToPythonKwargs("label", "sim"));
      ax.attr("legend")();
      if (subplot_count == 1) {
        ax.attr("set_title")("brick_revolute_x");
        ax.attr("set_ylabel")("angle (radians)");
      } else if (subplot_count == 2) {
        ax.attr("set_title")("brick_translate_y");
        ax.attr("set_ylabel")("y (m)");
      } else if (subplot_count == 3) {
        ax.attr("set_title")("brick_translate_z");
        ax.attr("set_ylabel")("z (m)");
        ax.attr("set_xlabel")("time (s)");
      }
      subplot_count++;
    }
    common::CallPython("show");
  }
  return 0;
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::logging::HandleSpdlogGflags();
  return drake::examples::planar_gripper::DoMain();
}
