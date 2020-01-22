#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/examples/planar_gripper/planar_gripper_lcm.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/examples/planar_gripper/planar_gripper.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/primitives/zero_order_hold.h"
#include "drake/examples/planar_gripper/finger_brick_control.h"
#include "drake/examples/planar_gripper/finger_brick.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/systems/lcm/connect_lcm_scope.h"

namespace drake {
namespace examples {
namespace planar_gripper {
namespace {

using multibody::RevoluteJoint;
using multibody::ContactResults;

DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_double(simulation_time, 2.75,
              "Desired duration of the simulation in seconds.");
DEFINE_double(time_step, 1e-3,
              "If greater than zero, the plant is modeled as a system with "
              "discrete updates and period equal to this time_step. "
              "If 0, the plant is modeled as a continuous system.");
DEFINE_double(penetration_allowance, 0.2,
              "The contact penetration allowance.");
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
// Initial finger joint angles.
// (for reference [-0.68, 1.21] sets the ftip at the center when box rot is
// zero)
DEFINE_int32(finger_to_control, 3, "Finger to control: {1, 2, 3}.");
DEFINE_string(contact_face, "NegZ",
              "The brick face to make contact with: {PosZ, NegZ, PosY, NegY}.");
DEFINE_double(f1_base, -0.15 /*-0.35*/, "f1_base");  // shoulder joint
DEFINE_double(f1_mid, 1.2 /* 1.3 */, "f1_mid");  // elbow joint
DEFINE_double(f2_base, -0.15, "f2_base");
DEFINE_double(f2_mid, 1.2 /* 0.84 */, "f2_mid");
DEFINE_double(f3_base, -0.15, "f3_base");
DEFINE_double(f3_mid, 1.2 /* 0.84 */, "f3_mid");
DEFINE_double(brick_thetadot0, 0, "initial brick rotational velocity.");
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
DEFINE_double(kd_base, 1.0, "joint damping for base joint.");
DEFINE_double(kd_mid, 1.0, "joint damping for mid joint.");
DEFINE_double(kpy, 0, "y-axis position gain (in brick frame).");
DEFINE_double(kdy, 0, "y-axis derivative gain (in brick frame).");
DEFINE_double(kpz, 0, "z-axis position gain (in brick frame).");
DEFINE_double(kdz, 15e3, "z-axis derivative gain (in brick frame).");
DEFINE_double(kpfy, 3e3, "y-axis proportional force gain (in brick frame).");
DEFINE_double(kpfz, 5e3, "z-axis proportional force gain (in brick frame).");
DEFINE_double(kify, 1e2, "y-axis integral force gain (in brick frame).");
DEFINE_double(kifz, 1e2, "z-axis integral force gain (in brick frame).");
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

// QP task parameters
// finger 1: theta0=-2.679793, thetaf=-1.308997
// finger 2: theta0=1.509, thetaf=2.87979
DEFINE_double(theta0, -M_PI_4 + 0.2, "initial theta (rad)");
DEFINE_double(thetaf, M_PI_4, "final theta (rad)");
DEFINE_double(T, 1.5, "time horizon (s)");

DEFINE_double(QP_Kp, 150 /* 50 */, "QP controller Kp gain");
DEFINE_double(QP_Kd, 20 /* 5 */, "QP controller Kd gain"); /* 20 for brick only */
DEFINE_double(QP_weight_thetaddot_error, 1, "thetaddot error weight.");
DEFINE_double(QP_weight_f_Cb_B, 1, "Contact force magnitude penalty weight");
DEFINE_double(QP_mu, 1.0, "QP mu");  /* MBP defaults to mu1 == mu2 == 1.0 */
// TODO(rcory) Pass in QP_mu to brick and fingertip-sphere collision geoms.

DEFINE_bool(assume_zero_brick_damping, false,
            "Override brick joint damping with zero.");

// TODO(rcory) Deprecate this.
DEFINE_string(QPType, "gripper", "The type of QP to use {finger, gripper}");

void GetControllerOptions(const PlanarGripper& planar_gripper,
                          ForceControlOptions* foptions,
                          QPControlOptions* qpoptions) {
  double brick_damping = 0;
  auto& plant = planar_gripper.get_multibody_plant();
  if (!FLAGS_assume_zero_brick_damping) {
    brick_damping =
        plant.GetJointByName<RevoluteJoint>("brick_revolute_x_joint")
            .damping();
  }
  double brick_inertia = dynamic_cast<const multibody::RigidBody<double>&>(
      plant.GetFrameByName("brick_link").body())
      .default_rotational_inertia()
      .get_moments()(0);

  foptions->kpfy_ = FLAGS_kpfy;
  foptions->kpfz_ = FLAGS_kpfz;
  foptions->kify_ = FLAGS_kify;
  foptions->kifz_ = FLAGS_kifz;
  foptions->kpy_ = FLAGS_kpy;
  foptions->kdy_ = FLAGS_kdy;
  foptions->kpz_ = FLAGS_kpz;
  foptions->kdz_ = FLAGS_kdz;
  foptions->Kd_ << FLAGS_kd_base, 0, 0, FLAGS_kd_mid;
  foptions->K_compliance_ = FLAGS_K_compliance;
  foptions->D_damping_ = FLAGS_D_damping;
  foptions->brick_damping_ = brick_damping;
  foptions->brick_inertia_ = brick_inertia;
  foptions->always_direct_force_control_ = FLAGS_always_direct_force_control;
  if (FLAGS_finger_to_control == 1) {
    foptions->finger_to_control_ = Finger::kFinger1;
  } else if (FLAGS_finger_to_control == 2) {
    foptions->finger_to_control_ = Finger::kFinger2;
  } else if (FLAGS_finger_to_control == 3) {
    foptions->finger_to_control_ = Finger::kFinger3;
  } else {
    throw std::logic_error("Undefined Finger specified.");
  }

  // Creates the QP controller and connects to the force controller.
  qpoptions->T_ = FLAGS_T;
  qpoptions->theta0_ = FLAGS_theta0;
  qpoptions->thetaf_ = FLAGS_thetaf;
  qpoptions->QP_Kp_ = FLAGS_QP_Kp;
  qpoptions->QP_Kd_ = FLAGS_QP_Kd;
  qpoptions->QP_weight_thetaddot_error_ = FLAGS_QP_weight_thetaddot_error;
  qpoptions->QP_weight_f_Cb_B_ = FLAGS_QP_weight_f_Cb_B;
  qpoptions->QP_mu_ = FLAGS_QP_mu;
  qpoptions->brick_only_ = FLAGS_brick_only;
  qpoptions->viz_force_scale_ = FLAGS_viz_force_scale;
  qpoptions->brick_damping_ = brick_damping;
  qpoptions->brick_inertia_ = brick_inertia;
  qpoptions->brick_type_ = BrickType::PinBrick;

  // Note: finger_face_assignments_ should only be used for brick only
  // simulation. However, currently I (rcory) don't have a good way of computing
  // the closest contact face for a given finger, so we instead use the
  // hardcoded values here (used in DoConnectGripperQPController).
  // TODO(rcory) Implement a proper "ComputeClosestBrickFace" method for
  //  finger/brick simulations.
  if (FLAGS_contact_face == "PosZ") {
    qpoptions->finger_face_assignments_.emplace(
        foptions->finger_to_control_,
        std::make_pair(BrickFace::kPosZ, Eigen::Vector2d(FLAGS_yc, 0.05)));
  } else if (FLAGS_contact_face == "NegZ") {
    qpoptions->finger_face_assignments_.emplace(
        foptions->finger_to_control_,
        std::make_pair(BrickFace::kNegZ, Eigen::Vector2d(FLAGS_yc, -0.05)));
  } else if (FLAGS_contact_face == "PosY") {
    qpoptions->finger_face_assignments_.emplace(
        foptions->finger_to_control_,
        std::make_pair(BrickFace::kPosY, Eigen::Vector2d(0.05, FLAGS_zc)));
  } else if (FLAGS_contact_face == "NegY") {
    qpoptions->finger_face_assignments_.emplace(
        foptions->finger_to_control_,
        std::make_pair(BrickFace::kNegY, Eigen::Vector2d(-0.05, FLAGS_zc)));
  } else {
    throw std::logic_error("Undefined contact face specified.");
  }
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
  planar_gripper->zero_gravity(FLAGS_zero_gravity);

  // Setup the 1-dof brick version of the plant.
  auto X_WG = math::RigidTransformd(
      math::RollPitchYaw<double>(FLAGS_G_ROT * M_PI / 180, 0, 0),
      Eigen::Vector3d::Zero());
  planar_gripper->set_X_WG(X_WG);
  planar_gripper->SetupPinBrick(FLAGS_orientation);
  planar_gripper->set_penetration_allowance(FLAGS_penetration_allowance);
  planar_gripper->set_stiction_tolerance(FLAGS_stiction_tolerance);

  // Finalize and build the diagram.
  planar_gripper->Finalize();

  lcm::DrakeLcm drake_lcm;
  systems::lcm::LcmInterfaceSystem* lcm =
      builder.AddSystem<systems::lcm::LcmInterfaceSystem>(&drake_lcm);

  // TODO(rcory) Uncomment once we have the QP planner under LCM.
//  auto planner_sub = builder.AddSystem(
//      systems::lcm::LcmSubscriberSystem::Make<
//          drake::lcmt_planar_gripper_plan>("PLANAR_GRIPPER_PLAN", lcm));
//  auto planner_decoder = builder.AddSystem<GripperCommandDecoder>();
//  builder.Connect(planner_sub->get_output_port(),
//                  planner_decoder->get_input_port(0));

  ForceControlOptions foptions; QPControlOptions qpoptions;
  GetControllerOptions(*planar_gripper, &foptions, &qpoptions);
  if (!FLAGS_brick_only) {
    ForceController* force_controller =
        SetupForceController(*planar_gripper, drake_lcm, foptions, &builder);
    std::unordered_map<Finger, ForceController&> finger_force_control_map;
    finger_force_control_map.emplace(foptions.finger_to_control_,
                                     *force_controller);
    if (FLAGS_QPType == "finger") {
      ConnectQPController(*planar_gripper, drake_lcm, finger_force_control_map,
                          QPType::FingerQP, qpoptions, &builder);
    } else if (FLAGS_QPType == "gripper") {
          ConnectQPController(*planar_gripper, drake_lcm,
          finger_force_control_map,
                              QPType::GripperQP, qpoptions, &builder);
    } else {
      throw std::logic_error("Unknown QPType flag");
    }
  } else {
    if (FLAGS_QPType == "finger") {
      ConnectQPController(*planar_gripper, drake_lcm, std::nullopt,
                          QPType::FingerQP, qpoptions, &builder);
    } else if (FLAGS_QPType == "gripper") {
    ConnectQPController(*planar_gripper, drake_lcm, std::nullopt,
                        QPType::GripperQP, qpoptions, &builder);
    } else {
      throw std::logic_error("Unknown QPType flag");
    }
  }

  // publish body frames.
  auto frame_viz = builder.AddSystem<FrameViz>(
      planar_gripper->get_multibody_plant(), drake_lcm, 1.0 / 60.0, true);
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
          "PLANAR_GRIPPER_STATUS", lcm, kGripperLcmStatusPeriod));
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

  auto diagram = builder.Build();

  // Set the initial conditions for the planar-gripper.
  std::map<std::string, double> init_gripper_pos_map;
  init_gripper_pos_map["finger1_BaseJoint"] = FLAGS_f1_base;
  init_gripper_pos_map["finger1_MidJoint"] = FLAGS_f1_mid;
  init_gripper_pos_map["finger2_BaseJoint"] = FLAGS_f2_base;
  init_gripper_pos_map["finger2_MidJoint"] = FLAGS_f2_mid;
  init_gripper_pos_map["finger3_BaseJoint"] = FLAGS_f3_base;
  init_gripper_pos_map["finger3_MidJoint"] = FLAGS_f3_mid;

  auto gripper_initial_positions = MakePositionVector(
      planar_gripper->get_control_plant(), init_gripper_pos_map);

  // Create a context for the diagram.
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  // If we are only simulating the brick, then ignore the force controller by
  // fixing the plant's actuation input port to zero.
  if (FLAGS_brick_only) {
    systems::Context<double>& planar_gripper_context =
        diagram->GetMutableSubsystemContext(*planar_gripper,
                                            diagram_context.get());
    Eigen::VectorXd tau_actuation = Eigen::VectorXd::Zero(kNumJoints);
    planar_gripper_context.FixInputPort(
        planar_gripper->GetInputPort("actuation").get_index(), tau_actuation);
  }

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  systems::Context<double>& simulator_context = simulator.get_mutable_context();

  // TODO(rcory) Uncomment this once we have the QP planner under LCM.
//  planner_decoder->set_initial_position(
//      &diagram->GetMutableSubsystemContext(*planner_decoder,
//                                           &simulator_context),
//      gripper_initial_positions);

  planar_gripper->SetGripperPosition(&simulator_context,
                                     gripper_initial_positions);
  planar_gripper->SetBrickPosition(simulator_context, Vector1d(FLAGS_theta0));

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