#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/examples/planar_gripper/finger_brick.h"
#include "drake/examples/planar_gripper/finger_brick_control.h"
#include "drake/examples/planar_gripper/planar_gripper.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/examples/planar_gripper/planar_gripper_utils.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_value_source.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/trajectory_source.h"

namespace drake {
namespace examples {
namespace planar_gripper {
namespace {

using multibody::ContactResults;
using multibody::RevoluteJoint;

// TODO(rcory) Move all common flags to a shared YAML file.
DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_double(simulation_time, 4.0,
              "Desired duration of the simulation in seconds.");
DEFINE_double(time_step, 1e-3,
              "If greater than zero, the plant is modeled as a system with "
              "discrete updates and period equal to this time_step. "
              "If 0, the plant is modeled as a continuous system.");
DEFINE_double(penetration_allowance, 0.2, "The contact penetration allowance.");
DEFINE_double(stiction_tolerance, 1e-3, "MBP v_stiction_tolerance");

// Initial finger angles.
DEFINE_double(f1_base, -0.55, "f1_base");  // shoulder joint
DEFINE_double(f1_mid, 1.5, "f1_mid");      // elbow joint
DEFINE_double(f2_base, 0.75, "f2_base");
DEFINE_double(f2_mid, -0.7, "f2_mid");
DEFINE_double(f3_base, -0.15, "f3_base");
DEFINE_double(f3_mid, 1.2, "f3_mid");

DEFINE_double(G_ROT, 0,
              "Rotation of gripper frame (G) w.r.t. the world frame W around "
              "the x-axis (deg).");

// Hybrid position/force control parameters.
DEFINE_double(kd_base_joint, 1.0, "joint damping for base joint.");
DEFINE_double(kd_mid_joint, 1.0, "joint damping for mid joint.");
DEFINE_double(kp_t, 0, "Tangential position gain (in brick frame).");
DEFINE_double(kd_t, 20e2, "Tangential derivative gain (in brick frame).");
DEFINE_double(kp_n, 0, "Normal position gain (in brick frame).");
DEFINE_double(kd_n, 15e3, "Normal derivative gain (in brick frame).");
DEFINE_double(kpf_t, 3e3,
              "Tangential proportional force gain (in brick frame).");
DEFINE_double(kpf_n, 5e3, "Normal proportional force gain (in brick frame).");
DEFINE_double(kif_t, 1e2, "Tangential integral force gain (in brick frame).");
DEFINE_double(kif_n, 1e2, "Normal integral force gain (in brick frame).");
DEFINE_double(K_compliance, 2e3, "Impedance control stiffness.");
DEFINE_double(D_damping, 1e3, "Impedance control damping.");

DEFINE_double(viz_force_scale, 1,
              "scale factor for visualizing spatial force arrow");

// Define which fingers are used for the brick rotation.
DEFINE_bool(use_finger1, true, "Use finger1?");
DEFINE_bool(use_finger2, true, "Use finger2?");
DEFINE_bool(use_finger3, true, "Use finger3?");

// Boundary conditions.
DEFINE_double(theta0, -M_PI_4 + 0.2, "initial theta (rad)");
DEFINE_double(thetadot0, 0, "initial brick rotational velocity.");
DEFINE_double(thetaf, M_PI_4, "final theta (rad)");
DEFINE_double(y0, 0.01, "initial brick y position (m).");
DEFINE_double(z0, 0, "initial brick z position (m).");
DEFINE_double(yf, 0, "final brick y position (m).");
DEFINE_double(zf, 0, "final brick z position (m).");
DEFINE_double(T, 1.5, "time horizon (s)");

// QP task parameters.
DEFINE_string(brick_type, "pinned",
              "Defines the brick type: {pinned, planar}.");
DEFINE_double(QP_plan_dt, 0.002, "The QP planner's timestep.");
DEFINE_double(QP_Kp_t, 350, "QP controller translational Kp gain.");
DEFINE_double(QP_Kd_t, 100, "QP controller translational Kd gain.");
DEFINE_double(QP_Kp_r_pinned, 150,
              "QP controller rotational Kp gain for pinned brick.");
DEFINE_double(QP_Kd_r_pinned, 50,
              "QP controller rotational Kd gain for pinned brick.");
DEFINE_double(QP_Kp_r_planar, 195,
              "QP controller rotational Kp gain for planar brick.");
DEFINE_double(QP_Kd_r_planar, 120,
              "QP controller rotational Kd gain for planar brick.");
DEFINE_double(QP_weight_thetaddot_error, 1, "thetaddot error weight.");
DEFINE_double(QP_weight_a_error, 1, "translational acceleration error weight.");
DEFINE_double(QP_weight_f_Cb_B, 1, "Contact force magnitude penalty weight");
DEFINE_double(QP_mu, 1.0, "QP mu"); /* MBP defaults to mu1 == mu2 == 1.0 */
// TODO(rcory) Pass in QP_mu to brick and fingertip-sphere collision geoms.

DEFINE_bool(test, false,
            "If true, checks the simulation result against a known value.");
DEFINE_double(switch_time, 0.7, "Hybrid control switch time");
DEFINE_bool(print_keyframes, false,
            "Print joint positions (keyframes) to standard out?");

// For this demo, we switch from position control to torque control after a
// certain time has elapsed (given by the flag switch_time). This should occur
// right after contact is initiated with all three fingers.
// This system has a single output port that contains the control type to use
// based on the current context time.
class OutputControlType final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(OutputControlType);

  OutputControlType() {
    this->DeclareAbstractOutputPort("control_type",
                                    &OutputControlType::SetOutput);
  }
  void SetOutput(const systems::Context<double>& context,
                 ControlType* control_type) const {
    if (context.get_time() > FLAGS_switch_time) {
      *control_type = ControlType::kTorque;
    } else {
      *control_type = ControlType::kPosition;
    }
  }
};

/// Utility function that returns the fingers that will be used for this
/// simulation.
std::unordered_set<Finger> FingersToControl() {
  std::unordered_set<Finger> fingers;
  if (FLAGS_use_finger1) {
    fingers.emplace(Finger::kFinger1);
  }
  if (FLAGS_use_finger2) {
    fingers.emplace(Finger::kFinger2);
  }
  if (FLAGS_use_finger3) {
    fingers.emplace(Finger::kFinger3);
  }
  return fingers;
}

void GetQPPlannerOptions(const PlanarGripper& planar_gripper,
                         const BrickType& brick_type,
                         QPControlOptions* qpoptions) {
  double brick_rotational_damping = planar_gripper.GetBrickPinJointDamping();

  // Get the brick's Ixx moment of inertia (i.e., around the pinned axis).
  const int kIxx_index = 0;
  double brick_inertia = planar_gripper.GetBrickMoments()(kIxx_index);
  double brick_mass = planar_gripper.GetBrickMass();

  qpoptions->T_ = FLAGS_T;
  qpoptions->plan_dt = FLAGS_QP_plan_dt;
  qpoptions->yf_ = FLAGS_yf;
  qpoptions->zf_ = FLAGS_zf;
  qpoptions->thetaf_ = FLAGS_thetaf;
  qpoptions->QP_Kp_r_ =
      (brick_type == BrickType::PinBrick ? FLAGS_QP_Kp_r_pinned
                                         : FLAGS_QP_Kp_r_planar);
  qpoptions->QP_Kd_r_ =
      (brick_type == BrickType::PinBrick ? FLAGS_QP_Kd_r_pinned
                                         : FLAGS_QP_Kd_r_planar);
  qpoptions->QP_Kp_t_ =
      Eigen::Vector2d(FLAGS_QP_Kp_t, FLAGS_QP_Kp_t).asDiagonal();
  qpoptions->QP_Kd_t_ =
      Eigen::Vector2d(FLAGS_QP_Kd_t, FLAGS_QP_Kd_t).asDiagonal();
  qpoptions->QP_weight_thetaddot_error_ = FLAGS_QP_weight_thetaddot_error;
  qpoptions->QP_weight_acceleration_error_ = FLAGS_QP_weight_a_error;
  qpoptions->QP_weight_f_Cb_B_ = FLAGS_QP_weight_f_Cb_B;
  qpoptions->QP_mu_ = FLAGS_QP_mu;
  qpoptions->brick_only_ = false;
  qpoptions->viz_force_scale_ = FLAGS_viz_force_scale;
  qpoptions->brick_rotational_damping_ = brick_rotational_damping;
  qpoptions->brick_translational_damping_ = 0;
  qpoptions->brick_inertia_ = brick_inertia;
  qpoptions->brick_mass_ = brick_mass;
  qpoptions->brick_spatial_force_assignments_ =
      BrickSpatialForceAssignments(FingersToControl());
  qpoptions->brick_type_ = brick_type;
}

void GetForceControllerOptions(const PlanarGripper& planar_gripper,
                               const Finger finger,
                               ForceControlOptions* foptions) {
  foptions->kpf_t_ = FLAGS_kpf_t;
  foptions->kpf_n_ = FLAGS_kpf_n;
  foptions->kif_t_ = FLAGS_kif_t;
  foptions->kif_n_ = FLAGS_kif_n;
  foptions->kp_t_ = FLAGS_kp_t;
  foptions->kd_t_ = FLAGS_kd_t;
  foptions->kp_n_ = FLAGS_kp_n;
  foptions->kd_n_ = FLAGS_kd_n;
  foptions->Kd_joint_ << FLAGS_kd_base_joint, 0, 0, FLAGS_kd_mid_joint;
  foptions->K_compliance_ = FLAGS_K_compliance;
  foptions->D_damping_ = FLAGS_D_damping;
  foptions->always_direct_force_control_ = false;
  foptions->finger_to_control_ = finger;
}

int DoMain() {
  systems::DiagramBuilder<double> builder;
  auto planar_gripper = builder.AddSystem<PlanarGripper>(
      FLAGS_time_step, ControlType::kHybrid, false /* no floor */);

  // Set up the plant.
  auto X_WG = math::RigidTransformd(
      math::RollPitchYaw<double>(FLAGS_G_ROT * M_PI / 180, 0, 0),
      Eigen::Vector3d::Zero());
  planar_gripper->set_X_WG(X_WG);
  BrickType brick_type;
  if (FLAGS_brick_type == "pinned") {
    planar_gripper->SetupPinBrick("vertical");
    brick_type = BrickType::PinBrick;
  } else if (FLAGS_brick_type == "planar") {
    planar_gripper->SetupPlanarBrick("vertical");
    brick_type = BrickType::PlanarBrick;
  } else {
    throw std::runtime_error("Unknown BrickType.");
  }
  planar_gripper->set_penetration_allowance(FLAGS_penetration_allowance);
  planar_gripper->set_stiction_tolerance(FLAGS_stiction_tolerance);
  planar_gripper->zero_gravity();

  // Finalize and build the diagram.
  planar_gripper->Finalize();

  lcm::DrakeLcm drake_lcm;

  QPControlOptions qpoptions;
  GetQPPlannerOptions(*planar_gripper, brick_type, &qpoptions);

  std::unordered_map<Finger, ForceController&> finger_force_control_map;
  for (auto& finger : FingersToControl()) {
    ForceControlOptions foptions;
    GetForceControllerOptions(*planar_gripper, finger, &foptions);
    DRAKE_DEMAND(finger == foptions.finger_to_control_);
    ForceController* force_controller =
        SetupForceController(*planar_gripper, &drake_lcm, foptions, &builder);
    finger_force_control_map.emplace(finger, *force_controller);
  }

  // Parse the keyframes
  const std::string keyframe_path =
      "drake/examples/planar_gripper/pinned_brick_postures_03.txt";
  MatrixX<double> finger_keyframes;
  std::map<std::string, int> finger_joint_name_to_row_index_map;
  std::tie(finger_keyframes, finger_joint_name_to_row_index_map) =
      ParseKeyframes(keyframe_path);

  // Create the individual finger matrices. For this demo, we assume we have
  // three fingers, two joints each.
  DRAKE_DEMAND(kNumFingers == 3);
  DRAKE_DEMAND(kNumJointsPerFinger == 2);
  int num_keys = finger_keyframes.cols();

  // Creates the time vector for the plan interpolator.
  Eigen::VectorXd times = Eigen::VectorXd::Zero(finger_keyframes.cols());
  for (int i = 1; i < finger_keyframes.cols(); ++i) {
    times(i) = i * 0.1 /* plan dt */;
  }

  // Create and connect the trajectory sources.
  for (int i = 0; i < kNumFingers; i++) {
    MatrixX<double> fn_keyframes(2, num_keys);
    fn_keyframes.row(0) = finger_keyframes.row(
        finger_joint_name_to_row_index_map["finger" + std::to_string(i + 1) +
                                           "_BaseJoint"]);
    fn_keyframes.row(1) = finger_keyframes.row(
        finger_joint_name_to_row_index_map["finger" + std::to_string(i + 1) +
                                           "_MidJoint"]);
    const auto finger_pp =
        trajectories::PiecewisePolynomial<double>::CubicShapePreserving(
            times, fn_keyframes);
    const auto finger_state_src =
        builder.AddSystem<systems::TrajectorySource<double>>(
            finger_pp, 1 /* with one derivative */);
    builder.Connect(finger_state_src->get_output_port(),
                    planar_gripper->GetInputPort(
                        "finger" + std::to_string(i + 1) + "_desired_state"));
  }

  // Add and connect the control_type source system.
  auto control_type_src = builder.AddSystem<OutputControlType>();
  DRAKE_DEMAND(finger_force_control_map.size() == kNumFingers);
  for (auto iter = finger_force_control_map.begin();
       iter != finger_force_control_map.end(); ++iter) {
    Finger finger = iter->first;
    ForceController& force_controller = finger_force_control_map.at(finger);
    builder.Connect(
        force_controller.get_torque_output_port(),
        planar_gripper->GetInputPort(to_string(finger) + "_torque_control_u"));
    builder.Connect(
        control_type_src->GetOutputPort("control_type"),
        planar_gripper->GetInputPort(to_string(finger) + "_control_type"));
    }

    ConnectQPController(*planar_gripper, &drake_lcm, finger_force_control_map,
                        qpoptions, &builder);

    // publish body frames.
    auto frame_viz = builder.AddSystem<FrameViz>(
        planar_gripper->get_multibody_plant(), &drake_lcm, 1.0 / 60.0, false);
    builder.Connect(planar_gripper->GetOutputPort("plant_state"),
                    frame_viz->get_input_port(0));

    math::RigidTransformd goal_frame;
    goal_frame.set_rotation(math::RollPitchYaw<double>(FLAGS_thetaf, 0, 0));
    PublishFramesToLcm("GOAL_FRAME", {goal_frame}, {"goal"}, &drake_lcm);

    // Connect drake visualizer.
    geometry::ConnectDrakeVisualizer(
        &builder, planar_gripper->get_mutable_scene_graph(),
        planar_gripper->GetOutputPort("pose_bundle"), &drake_lcm);

    // Publish contact results for visualization.
    ConnectContactResultsToDrakeVisualizer(
        &builder, planar_gripper->get_mutable_multibody_plant(),
        planar_gripper->GetOutputPort("contact_results"));

    if (FLAGS_print_keyframes) {
      // Prints the state to standard output.
      auto joint_ordering = GetPreferredGripperJointOrdering();
      joint_ordering.emplace_back("brick_revolute_x_joint");
      auto state_pub = builder.AddSystem<PrintKeyframes>(
          planar_gripper->get_multibody_plant(), joint_ordering, 0.1,
          false /* don't print time */);
      builder.Connect(planar_gripper->GetOutputPort("plant_state"),
                      state_pub->GetInputPort("plant_state"));
    }

    auto diagram = builder.Build();

    // Set the initial conditions for the planar-gripper.
    std::map<std::string, double> init_gripper_pos_map;
    init_gripper_pos_map["finger1_BaseJoint"] = FLAGS_f1_base;
    init_gripper_pos_map["finger1_MidJoint"] = FLAGS_f1_mid;
    init_gripper_pos_map["finger2_BaseJoint"] = FLAGS_f2_base;
    init_gripper_pos_map["finger2_MidJoint"] = FLAGS_f2_mid;
    init_gripper_pos_map["finger3_BaseJoint"] = FLAGS_f3_base;
    init_gripper_pos_map["finger3_MidJoint"] = FLAGS_f3_mid;
    auto gripper_initial_positions =
        planar_gripper->MakeGripperPositionVector(init_gripper_pos_map);

    // Set the initial conditions for the brick.
    std::map<std::string, double> init_brick_pos_map;
    std::map<std::string, double> init_brick_vel_map;
    init_brick_pos_map["brick_revolute_x_joint"] = FLAGS_theta0;
    init_brick_vel_map["brick_revolute_x_joint"] = FLAGS_thetadot0;
    if (brick_type == BrickType::PlanarBrick) {
      init_brick_pos_map["brick_translate_y_joint"] = FLAGS_y0;
      init_brick_pos_map["brick_translate_z_joint"] = FLAGS_z0;
      init_brick_vel_map["brick_translate_y_joint"] = 0;
      init_brick_vel_map["brick_translate_z_joint"] = 0;
  }
  auto brick_initial_positions =
      planar_gripper->MakeBrickPositionVector(init_brick_pos_map);
  auto brick_initial_velocities =
      planar_gripper->MakeBrickVelocityVector(init_brick_vel_map);

  // Create a context for the diagram.
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& planar_gripper_context =
      diagram->GetMutableSubsystemContext(*planar_gripper,
                                          diagram_context.get());

  planar_gripper->SetGripperPosition(&planar_gripper_context,
                                     gripper_initial_positions);
  planar_gripper->SetGripperVelocity(&planar_gripper_context,
                                     Eigen::VectorXd::Zero(kNumGripperJoints));
  planar_gripper->SetBrickPosition(&planar_gripper_context,
                                   brick_initial_positions);
  planar_gripper->SetBrickVelocity(&planar_gripper_context,
                                   brick_initial_velocities);

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);

  // TODO(rcory) Implement a proper unit test once all shared parameters are
  //  moved to a YAML file.
  if (FLAGS_test) {
    VectorX<double> x_known(14);
    x_known << -0.6779083, 0.4575399, -0.7382065, 0.7823212, 0.6807362,
        -1.2508255, 1.1141098, 0.0041306, -0.0002022, 0.0002803, -0.0009766,
        -0.0082137, 0.0000462, -0.0012514;
    const auto& post_sim_context = simulator.get_context();
    const auto& post_plant_context = diagram->GetSubsystemContext(
        planar_gripper->get_mutable_multibody_plant(), post_sim_context);
    const auto post_plant_state =
        planar_gripper->get_multibody_plant().GetPositionsAndVelocities(
            post_plant_context);

    // Check to within an arbitrary threshold.
    DRAKE_DEMAND(x_known.isApprox(post_plant_state, 1e-6));
  }

  return 0;
}

}  // namespace
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("A simple planar gripper example.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::planar_gripper::DoMain();
}
