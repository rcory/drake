#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/examples/planar_gripper/finger_brick.h"
#include "drake/examples/planar_gripper/finger_brick_control.h"
#include "drake/examples/planar_gripper/planar_gripper.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/examples/planar_gripper/planar_gripper_lcm.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/connect_lcm_scope.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"

namespace drake {
namespace examples {
namespace planar_gripper {
namespace {

using multibody::ContactResults;
using multibody::RevoluteJoint;

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
DEFINE_double(floor_coef_static_friction, 0 /*0.5*/,
              "The floor's coefficient of static friction");
DEFINE_double(floor_coef_kinetic_friction, 0 /*0.5*/,
              "The floor's coefficient of kinetic friction");
DEFINE_double(brick_floor_penetration, 0 /* 1e-5 */,
              "Determines how much the brick should penetrate the floor "
              "(in meters). When simulating the vertical case this penetration "
              "distance will remain fixed.");
DEFINE_string(orientation, "vertical",
              "The orientation of the planar gripper. Options are {vertical, "
              "horizontal}.");
DEFINE_bool(visualize_contacts, true,
            "Visualize contacts in Drake visualizer.");

// Gripper/brick rotate specific flags
DEFINE_double(f1_base, -0.55, "f1_base");  // shoulder joint
DEFINE_double(f1_mid, 1.5, "f1_mid");      // elbow joint
DEFINE_double(f2_base, 0.75, "f2_base");
DEFINE_double(f2_mid, -0.7, "f2_mid");
DEFINE_double(f3_base, -0.15, "f3_base");
DEFINE_double(f3_mid, 1.2, "f3_mid");
DEFINE_double(G_ROT, 0,
              "Rotation of gripper frame (G) w.r.t. the world frame W around "
              "the x-axis (deg).");

// Note: The default plant sets up a vertical orientation with zero gravity.
// This is because the default penetration allowance is so high, that the
// brick penetrates the floor with gravity in the horizontal orientation.
// Setting orientation to vertical and eliminating gravity effectively
// achieves the same behavior as the existing finger/brick rotate.
// We throw if we try to setup a horizontal orientation with gravity on (for
// now).
DEFINE_bool(zero_gravity, true, "Always zero gravity?");

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
DEFINE_bool(always_direct_force_control, false,
            "Always use direct force control (i.e., no impedance control for "
            "regulating fingertip back to contact)?");
DEFINE_double(viz_force_scale, 1,
              "scale factor for visualizing spatial force arrow");
DEFINE_bool(brick_only, false, "Only simulate brick (no finger).");
DEFINE_double(
    yc, 0,
    "Value of y-coordinate offset for z-face contact (for brick-only sim).");
DEFINE_double(
    zc, 0,
    "Value of z-coordinate offset for y-face contact (for brick-only sim.");

// Define which fingers are used for the brick rotation.
DEFINE_bool(use_finger1, true, "Use finger1?");
DEFINE_bool(use_finger2, true, "Use finger2?");
DEFINE_bool(use_finger3, true, "Use finger3?");

// Boundary conditions.
DEFINE_double(theta0, -M_PI_4 + 0.2, "initial theta (rad)");
DEFINE_double(thetadot0, 0, "initial brick rotational velocity.");
DEFINE_double(thetaf, M_PI_4, "final theta (rad)");
DEFINE_double(y0, 0.01, "initial brick y position (m).");
DEFINE_double(z0, 0, "initial brick y position (m).");
DEFINE_double(yf, 0, "final brick y position (m).");
DEFINE_double(zf, 0, "final brick y position (m).");
DEFINE_double(T, 1.5, "time horizon (s)");

// QP task parameters.
DEFINE_string(brick_type, "pinned",
              "Defines the brick type: {pinned, planar}.");
DEFINE_double(QP_plan_dt, 0.002, "The QP planner's timestep.");

DEFINE_string(use_QP, "local",
              "We provide 3 types of QP controller, LCM, UDP or local.");
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

DEFINE_bool(assume_zero_brick_damping, false,
            "Override brick joint damping with zero.");

DEFINE_int32(publisher_local_port, 1103, "local port number for UDP publisher");
// publisher_remote_port should be the same as the receiver_local_port in
// run_planar_gripper_qp_udp_controller.
DEFINE_int32(publisher_remote_port, 1102,
             "remote port number for UDP publisher");
// I convert the IP address of my computer to unsigned long through
// https://www.smartconversion.com/unit_conversion/IP_Address_Converter.aspx
DEFINE_uint64(publisher_remote_address, 0,
              "remote IP address for UDP publisher.");
// receiver_local_port should be the same as the publisher_remote_port in
// run_planar_gripper_qp_udp_controller.
DEFINE_int32(receiver_local_port, 1100, "local port number for UDP receiver");

/// Note: finger_face_assignments_ should only be used for brick only
/// simulation. However, currently I (rcory) don't have a good way of computing
/// the closest contact face for a given finger, so we instead use the
/// hardcoded values here (used in DoConnectGripperQPController).
/// Note: The 2d contact position vector below is strictly for brick-only sim.
// TODO(rcory) Implement a proper "ComputeClosestBrickFace" method for
//  finger/brick simulations.
std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>
GetFingerFaceAssignments() {
  std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>
      finger_face_assignments;
  if (FLAGS_use_finger1 || FLAGS_brick_only) {
    finger_face_assignments.emplace(
        Finger::kFinger1,
        std::make_pair(BrickFace::kNegY, Eigen::Vector2d(-0.05, FLAGS_zc)));
  }
  if (FLAGS_use_finger2 || FLAGS_brick_only) {
    finger_face_assignments.emplace(
        Finger::kFinger2,
        std::make_pair(BrickFace::kPosY, Eigen::Vector2d(0.05, FLAGS_zc)));
  }
  if (FLAGS_use_finger3 || FLAGS_brick_only) {
    finger_face_assignments.emplace(
        Finger::kFinger3,
        std::make_pair(BrickFace::kNegZ, Eigen::Vector2d(FLAGS_yc, -0.05)));
  }
  return finger_face_assignments;
}

void GetQPPlannerOptions(const PlanarGripper& planar_gripper,
                         const BrickType& brick_type,
                         QPControlOptions* qpoptions) {
  double brick_rotational_damping = 0;
  if (!FLAGS_assume_zero_brick_damping) {
    brick_rotational_damping = planar_gripper.GetBrickPinJointDamping();
  }
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
  qpoptions->brick_only_ = FLAGS_brick_only;
  qpoptions->viz_force_scale_ = FLAGS_viz_force_scale;
  qpoptions->brick_rotational_damping_ = brick_rotational_damping;
  qpoptions->brick_translational_damping_ = 0;
  qpoptions->brick_inertia_ = brick_inertia;
  qpoptions->brick_mass_ = brick_mass;
  qpoptions->finger_face_assignments_ = GetFingerFaceAssignments();
  qpoptions->brick_type_ = brick_type;
}

void GetForceControllerOptions(const PlanarGripper& planar_gripper,
                               const Finger finger, const BrickFace brick_face,
                               ForceControlOptions* foptions) {
  double brick_damping = 0;
  if (!FLAGS_assume_zero_brick_damping) {
    brick_damping = planar_gripper.GetBrickPinJointDamping();
  }
  // Get the brick's Ixx moment of inertia (i.e., around the pinned axis).
  const int kIxx_index = 0;
  double brick_inertia = planar_gripper.GetBrickMoments()(kIxx_index);

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
  foptions->brick_damping_ = brick_damping;
  foptions->brick_inertia_ = brick_inertia;
  foptions->always_direct_force_control_ = FLAGS_always_direct_force_control;
  foptions->finger_to_control_ = finger;
  foptions->brick_face_ = brick_face;
}

int DoMain() {
  if (FLAGS_orientation == "horizontal" && !FLAGS_zero_gravity) {
    throw std::runtime_error(
        "Cannot setup horizontal gripper with gravity on, due to high "
        "penetration allowance. Instead setup a vertical gripper with zero "
        "gravity (for now) for a similar effect.");
  }

  systems::DiagramBuilder<double> builder;
  auto planar_gripper = builder.AddSystem<PlanarGripper>(
      FLAGS_time_step, false /* no position control */);

  // Set some plant parameters.
  planar_gripper->set_floor_coef_static_friction(
      FLAGS_floor_coef_static_friction);
  planar_gripper->set_floor_coef_kinetic_friction(
      FLAGS_floor_coef_kinetic_friction);
  planar_gripper->set_brick_floor_penetration(FLAGS_brick_floor_penetration);
  if (FLAGS_zero_gravity) {
    planar_gripper->zero_gravity();
  }

  // Setup the 1-dof brick version of the plant.
  auto X_WG = math::RigidTransformd(
      math::RollPitchYaw<double>(FLAGS_G_ROT * M_PI / 180, 0, 0),
      Eigen::Vector3d::Zero());
  planar_gripper->set_X_WG(X_WG);
  BrickType brick_type;
  if (FLAGS_brick_type == "pinned") {
    planar_gripper->SetupPinBrick(FLAGS_orientation);
    brick_type = BrickType::PinBrick;
  } else if (FLAGS_brick_type == "planar") {
    planar_gripper->SetupPlanarBrick(FLAGS_orientation);
    brick_type = BrickType::PlanarBrick;
  } else {
    throw std::runtime_error("Unknown BrickType.");
  }
  planar_gripper->set_penetration_allowance(FLAGS_penetration_allowance);
  planar_gripper->set_stiction_tolerance(FLAGS_stiction_tolerance);

  // Finalize and build the diagram.
  planar_gripper->Finalize();

  lcm::DrakeLcm drake_lcm;
  systems::lcm::LcmInterfaceSystem* lcm =
      builder.AddSystem<systems::lcm::LcmInterfaceSystem>(&drake_lcm);

  QPControlOptions qpoptions;
  GetQPPlannerOptions(*planar_gripper, brick_type, &qpoptions);
  if (FLAGS_brick_only) {
    if (FLAGS_use_QP == "LCM") {
      ConnectLCMQPController(*planar_gripper, &drake_lcm, std::nullopt,
                             qpoptions, &builder);
    } else if (FLAGS_use_QP == "UDP") {
      ConnectUDPQPController(
          *planar_gripper, &drake_lcm, std::nullopt, qpoptions,
          FLAGS_publisher_local_port, FLAGS_publisher_remote_port,
          FLAGS_publisher_remote_address, FLAGS_receiver_local_port, &builder);
    } else if (FLAGS_use_QP == "local") {
      ConnectQPController(*planar_gripper, &drake_lcm, std::nullopt, qpoptions,
                          &builder);
    } else {
      throw std::runtime_error("use_QP must be either LCM, UDP or local");
    }
  } else {
    std::unordered_map<Finger, ForceController&> finger_force_control_map;
    for (auto& finger_face_assignment : qpoptions.finger_face_assignments_) {
      ForceControlOptions foptions;
      Finger finger = finger_face_assignment.first;
      BrickFace brick_face = finger_face_assignment.second.first;
      GetForceControllerOptions(*planar_gripper, finger, brick_face, &foptions);
      DRAKE_DEMAND(finger == foptions.finger_to_control_);
      DRAKE_DEMAND(brick_face == foptions.brick_face_);
      ForceController* force_controller =
          SetupForceController(*planar_gripper, &drake_lcm, foptions, &builder);
      finger_force_control_map.emplace(finger, *force_controller);
    }
    auto force_controllers_to_plant =
        builder.AddSystem<ForceControllersToPlantActuationMap>(
            planar_gripper->get_multibody_plant(), finger_force_control_map);
    force_controllers_to_plant->ConnectForceControllersToPlant(*planar_gripper,
                                                               &builder);
    if (FLAGS_use_QP == "LCM") {
      ConnectLCMQPController(*planar_gripper, &drake_lcm,
                             finger_force_control_map, qpoptions, &builder);
    } else if (FLAGS_use_QP == "UDP") {
      ConnectUDPQPController(
          *planar_gripper, &drake_lcm, finger_force_control_map, qpoptions,
          FLAGS_publisher_local_port, FLAGS_publisher_remote_port,
          FLAGS_publisher_remote_address, FLAGS_receiver_local_port, &builder);
    } else if (FLAGS_use_QP == "local") {
      ConnectQPController(*planar_gripper, &drake_lcm, finger_force_control_map,
                          qpoptions, &builder);
    } else {
      throw std::runtime_error("use_QP should be either LCM, UDP or local.");
    }
  }

  // publish body frames.
  auto frame_viz = builder.AddSystem<FrameViz>(
      planar_gripper->get_multibody_plant(), &drake_lcm, 1.0 / 60.0, true);
  builder.Connect(planar_gripper->GetOutputPort("plant_state"),
                  frame_viz->get_input_port(0));

  math::RigidTransformd goal_frame;
  goal_frame.set_rotation(math::RollPitchYaw<double>(FLAGS_thetaf, 0, 0));
  PublishFramesToLcm("GOAL_FRAME", {goal_frame}, {"goal"}, &drake_lcm);

  // Connect drake visualizer.
  geometry::ConnectDrakeVisualizer(
      &builder, planar_gripper->get_mutable_scene_graph(),
      planar_gripper->GetOutputPort("pose_bundle"));

  // Publish contact results for visualization.
  if (FLAGS_visualize_contacts) {
    ConnectContactResultsToDrakeVisualizer(
        &builder, planar_gripper->get_mutable_multibody_plant(),
        planar_gripper->GetOutputPort("contact_results"));
  }

  // Publish planar gripper status via LCM.
  auto status_pub = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<drake::lcmt_planar_gripper_status>(
          "PLANAR_GRIPPER_STATUS", lcm, kGripperLcmPeriod));
  auto status_encoder = builder.AddSystem<GripperStatusEncoder>();
  auto state_remapper = builder.AddSystem<MapStateToUserOrderedState>(
      planar_gripper->get_multibody_plant(),
      GetPreferredGripperStateOrdering());
  builder.Connect(planar_gripper->GetOutputPort("plant_state"),
                  state_remapper->get_input_port(0));
  builder.Connect(state_remapper->get_output_port(0),
                  status_encoder->get_state_input_port());
  builder.Connect(planar_gripper->GetOutputPort("force_sensor"),
                  status_encoder->get_force_input_port());
  builder.Connect(status_encoder->get_output_port(0),
                  status_pub->get_input_port());

  systems::lcm::ConnectLcmScope(planar_gripper->GetOutputPort("brick_state"),
                                "BRICK_STATE", &builder, lcm);

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

  // If we are only simulating the brick, then ignore the force controller by
  // fixing the plant's actuation input port to zero.
  if (FLAGS_brick_only) {
    Eigen::VectorXd tau_actuation = Eigen::VectorXd::Zero(kNumGripperJoints);
    planar_gripper_context.FixInputPort(
        planar_gripper->GetInputPort("actuation").get_index(), tau_actuation);
  }

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
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
  gflags::SetUsageMessage("A simple planar gripper example.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::planar_gripper::DoMain();
}
