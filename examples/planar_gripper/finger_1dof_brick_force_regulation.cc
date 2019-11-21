#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/systems/primitives/zero_order_hold.h"
#include "drake/systems/primitives/demultiplexer.h"
#include "drake/examples/planar_gripper/brick_qp.h"
#include "drake/examples/planar_gripper/planar_finger_qp.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/examples/planar_gripper/finger_brick.h"
#include "drake/examples/planar_gripper/finger_brick_control.h"

namespace drake {
namespace examples {
namespace planar_gripper {
namespace {

using geometry::SceneGraph;
using lcm::DrakeLcm;
using multibody::MultibodyPlant;
using multibody::Parser;
using multibody::RevoluteJoint;
using multibody::PrismaticJoint;
using multibody::ConnectContactResultsToDrakeVisualizer;
using multibody::ContactResults;

DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_double(simulation_time, 1.5,
              "Desired duration of the simulation in seconds.");
DEFINE_double(time_step, 1e-4,
            "If greater than zero, the plant is modeled as a system with "
            "discrete updates and period equal to this time_step. "
            "If 0, the plant is modeled as a continuous system.");

DEFINE_double(penetration_allowance, 0.2, "Penetration allowance.");
DEFINE_double(stiction_tolerance, 1e-3, "MBP v_stiction_tolerance");
DEFINE_double(gravity_accel, -9.81, "The acceleration due to gravity.");

// Initial finger joint angles.
// (for reference [-0.68, 1.21] sets the ftip at the center when box rot is
// zero)
DEFINE_double(j1, -0.15, "j1");  // shoulder joint
DEFINE_double(j2, 1.2 /* 0.84 */, "j2");  // elbow joint
DEFINE_double(brick_thetadot0, 0, "initial brick rotational velocity.");

// Hybrid position/force control paramters.
DEFINE_double(kd_j1, 0.2, "joint damping for joint 1.");
DEFINE_double(kd_j2, 0.2, "joint damping for joint 2.");
DEFINE_double(kpy, 0, "y-axis position gain (in brick frame).");
DEFINE_double(kdy, 0, "y-axis derivative gain (in brick frame).");
DEFINE_double(kpz, 0, "z-axis position gain (in brick frame).");
DEFINE_double(kdz, 15e3, "z-axis derivative gain (in brick frame).");
DEFINE_double(kfy, 25e3, "y-axis force gain (in brick frame).");
DEFINE_double(kfz, 20e3, "z-axis force gain (in brick frame).");
DEFINE_double(K_compliance, 10e3, "Impedance control stiffness.");
DEFINE_double(D_damping, 1e3, "Impedance control damping.");
DEFINE_bool(always_direct_force_control, false,
            "Always use direct force control (i.e., no impedance control for "
            "regulating fingertip back to contact)?");
DEFINE_double(viz_force_scale, 5,
              "scale factor for visualizing spatial force arrow");
DEFINE_bool(brick_only, false, "Only simulate brick (no finger).");

DEFINE_double(yc, 0,
              "y_Br contact point location for brick only sim.");
DEFINE_double(zc, 0.046,
              "z_br contact point location for brick only sim.");

// QP task parameters
DEFINE_double(theta0, -M_PI_4 + 0.2, "initial theta (rad)");
DEFINE_double(thetaf, M_PI_4, "final theta (rad)");
DEFINE_double(T, 1.5, "time horizon (s)");

DEFINE_double(QP_Kp, 60 /* 50 */, "QP controller Kp gain");
DEFINE_double(QP_Kd, 0 /* 5 */, "QP controller Kd gain");
DEFINE_double(QP_weight_thetaddot_error, 1, "thetaddot error weight.");
DEFINE_double(QP_weight_f_Cb_B, 1, "Contact force magnitued penalty weight");
DEFINE_double(QP_mu, 1.0, "QP mu");  /* MBP defaults to mu1 == mu2 == 1.0 */
// TODO(rcory) Pass in QP_mu to brick and fingertip-sphere collision geoms.

DEFINE_bool(assume_zero_brick_damping, false, "Override brick joint damping with zero.");


void PrintJointOrdering(const MultibodyPlant<double>& plant) {
  for (int i = 0; i < plant.num_joints(); i++) {
    auto& joint = plant.get_joint(multibody::JointIndex(i));
    drake::log()->info("Joint[{}]: {}", i, joint.name());
  }
}

int do_main() {
  systems::DiagramBuilder<double> builder;

  SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
  scene_graph.set_name("scene_graph");

  // Make and add the planar_gripper model.
  const std::string full_name = FindResourceOrThrow(
      "drake/examples/planar_gripper/planar_finger.sdf");
  MultibodyPlant<double>& plant =
      *builder.AddSystem<MultibodyPlant>(FLAGS_time_step);
  auto finger_index =
      Parser(&plant, &scene_graph).AddModelFromFile(full_name);
  if (FLAGS_brick_only) {
    WeldFingerFrame<double>(&plant, -1);
  } else {
    WeldFingerFrame<double>(&plant);
  }

  //   Adds the object to be manipulated.
  auto object_file_name =
      FindResourceOrThrow("drake/examples/planar_gripper/1dof_brick.sdf");
  auto brick_index =
      Parser(&plant, &scene_graph).AddModelFromFile(object_file_name, "object");

  // Add gravity
  Vector3<double> gravity(0, 0, FLAGS_gravity_accel);
  plant.mutable_gravity_field().set_gravity_vector(gravity);

  // Now the model is complete.
  plant.Finalize();

  // Set the penetration allowance for the simulation plant only
  plant.set_penetration_allowance(FLAGS_penetration_allowance);
  plant.set_stiction_tolerance(FLAGS_stiction_tolerance);

  // Sanity check on the availability of the optional source id before using it.
  DRAKE_DEMAND(plant.geometry_source_is_registered());

  // Extract the joint damping and rotational inertia parameters for the 1-dof
  // brick (used in QP and force controllers).
  double brick_damping = 0;
  if (!FLAGS_assume_zero_brick_damping) {
    brick_damping =
        plant.GetJointByName<multibody::RevoluteJoint>("brick_pin_joint")
            .damping();
  }
  double brick_inertia = dynamic_cast<const multibody::RigidBody<double>&>(
                             plant.GetFrameByName("brick_base").body())
                             .default_rotational_inertia()
                             .get_moments()(0);

  // Connect the force controller
  auto zoh_contact_results = builder.AddSystem<systems::ZeroOrderHold<double>>(
      1e-3, Value<ContactResults<double>>());

  auto zoh_joint_accels = builder.AddSystem<systems::ZeroOrderHold<double>>(
      1e-3, plant.num_velocities());

  // Setup the force controller.
  ForceControlOptions foptions;
  foptions.kfy_ = FLAGS_kfy;
  foptions.kfz_ = FLAGS_kfz;
  foptions.kpy_ = FLAGS_kpy;
  foptions.kdy_ = FLAGS_kdy;
  foptions.kpz_ = FLAGS_kpz;
  foptions.kdz_ = FLAGS_kdz;
  foptions.Kd_ << FLAGS_kd_j1, 0, 0, FLAGS_kd_j2;
  foptions.K_compliance_ = FLAGS_K_compliance;
  foptions.D_damping_ = FLAGS_D_damping;
  foptions.brick_damping_ = brick_damping;
  foptions.brick_inertia_ = brick_inertia;
  foptions.always_direct_force_control_ = FLAGS_always_direct_force_control;

  auto force_controller =
      builder.AddSystem<ForceController>(plant, scene_graph, foptions);
  builder.Connect(plant.get_state_output_port(finger_index),
                  force_controller->get_finger_state_actual_input_port());
  builder.Connect(plant.get_state_output_port(brick_index),
                  force_controller->get_brick_state_actual_input_port());
  builder.Connect(plant.get_contact_results_output_port(),
                  zoh_contact_results->get_input_port());
  builder.Connect(zoh_contact_results->get_output_port(),
                  force_controller->get_contact_results_input_port());
  builder.Connect(plant.get_joint_accelerations_output_port(),
                  zoh_joint_accels->get_input_port());
  builder.Connect(zoh_joint_accels->get_output_port(),
                  force_controller->get_accelerations_actual_input_port());
  builder.Connect(scene_graph.get_query_output_port(),
                  force_controller->get_geometry_query_input_port());

  // aux debugging info
  std::vector<int> sizes = {2, 2, 1}; // tau_des, f_des, ytip
  auto demux = builder.AddSystem<systems::Demultiplexer<double>>(sizes);
  builder.Connect(force_controller->get_torque_output_port(),
                  demux->get_input_port(0));
  builder.Connect(demux->get_output_port(0),
                  plant.get_actuation_input_port(finger_index));

  // We don't regulate position for now (set these to zero).
  // 6-vector represents pos-vel for fingertip contact point x-y-z. The control
  // ignores the x-components.
  const Vector6<double> tip_state_des_vec = Vector6<double>::Zero();
  auto const_pos_src =
      builder.AddSystem<systems::ConstantVectorSource>(tip_state_des_vec);
  builder.Connect(const_pos_src->get_output_port(),
                  force_controller->get_tip_state_desired_input_port());

  // Connect MBP snd SG.
  builder.Connect(
      plant.get_geometry_poses_output_port(),
      scene_graph.get_source_pose_port(plant.get_source_id().value()));
  builder.Connect(scene_graph.get_query_output_port(),
      plant.get_geometry_query_input_port());

  DrakeLcm lcm;
  geometry::ConnectDrakeVisualizer(&builder, scene_graph, &lcm);

  // Publish contact results for visualization.
  ConnectContactResultsToDrakeVisualizer(&builder, plant, &lcm);

  // Creates the QP controller and connects to the force controller.
  QPControlOptions qpoptions;
  qpoptions.T_ = FLAGS_T;
  qpoptions.theta0_ = FLAGS_theta0;
  qpoptions.thetaf_ = FLAGS_thetaf;
  qpoptions.QP_Kp_ = FLAGS_QP_Kp;
  qpoptions.QP_Kd_ = FLAGS_QP_Kd;
  qpoptions.QP_weight_thetaddot_error_ = FLAGS_QP_weight_thetaddot_error;
  qpoptions.QP_weight_f_Cb_B_ = FLAGS_QP_weight_f_Cb_B;
  qpoptions.QP_mu_ = FLAGS_QP_mu;
  qpoptions.brick_only_ = FLAGS_brick_only;
  qpoptions.viz_force_scale_ = FLAGS_viz_force_scale;
  qpoptions.yc_ = FLAGS_yc;
  qpoptions.zc_ = FLAGS_zc;
  qpoptions.brick_damping_ = brick_damping;
  qpoptions.brick_inertia_ = brick_inertia;
  ConnectControllers(plant, scene_graph, lcm, *force_controller, brick_index,
                     qpoptions, &builder);

  // publish body frames.
  auto frame_viz = builder.AddSystem<FrameViz>(plant, lcm, 1.0 / 30.0, true);
  builder.Connect(plant.get_state_output_port(), frame_viz->get_input_port(0));

  auto diagram = builder.Build();

  // Create a context for this system:
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  if (FLAGS_brick_only) {
    Eigen::Vector2d tau_d = Eigen::Vector2d::Zero();
    plant_context.FixInputPort(
        plant.get_actuation_input_port().get_index(), tau_d);
  }

  // Set finger initial conditions.
  VectorX<double> finger_initial_conditions = VectorX<double>::Zero(4);
  finger_initial_conditions << FLAGS_j1, FLAGS_j2, 0, 0;
  const RevoluteJoint<double>& sh_pin1 =
      plant.GetJointByName<RevoluteJoint>("finger_BaseJoint");
  const RevoluteJoint<double>& el_pin1 =
      plant.GetJointByName<RevoluteJoint>("finger_MidJoint");
  sh_pin1.set_angle(&plant_context, finger_initial_conditions(0));
  el_pin1.set_angle(&plant_context, finger_initial_conditions(1));

  // Set the brick's initial condition.
  plant.SetPositions(&plant_context, brick_index, Vector1d(FLAGS_theta0));
  plant.SetVelocities(&plant_context, brick_index,
                      Vector1d(FLAGS_brick_thetadot0));

  PrintJointOrdering(plant);

  math::RigidTransformd goal_frame;
  goal_frame.set_rotation(math::RollPitchYaw<double>(FLAGS_thetaf, 0, 0));
  PublishFramesToLcm("GOAL_FRAME", {goal_frame}, {"goal"}, &lcm);

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
  return drake::examples::planar_gripper::do_main();
}
