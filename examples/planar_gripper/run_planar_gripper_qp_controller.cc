/// @file
///
/// Implements an LCM based finger/brick QP controller.
// TODO(rcory) This file is still a WIP.

#include <gflags/gflags.h>

#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/examples/planar_gripper/planar_gripper_lcm.h"
#include "drake/examples/planar_gripper/planar_gripper.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/examples/planar_gripper/finger_brick_control.h"
#include "drake/lcmt_planar_plant_state.hpp"

namespace drake {
namespace examples {
namespace planar_gripper {
namespace {

DEFINE_double(simulation_time, 3.0, "Amount of time to simulate.");
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
DEFINE_double(theta0, -M_PI_4 + 0.2, "initial theta (rad)");
DEFINE_double(thetaf, M_PI_4, "final theta (rad)");
DEFINE_double(T, 1.5, "time horizon (s)");
DEFINE_double(QP_plan_dt, 0.002, "The QP planner's timestep.");

DEFINE_double(QP_Kp_ro, 150, "QP controller rotational Kp gain");
DEFINE_double(QP_Kd_ro, 50, "QP controller rotational Kd gain");
DEFINE_double(QP_weight_thetaddot_error, 1, "thetaddot error weight.");
DEFINE_double(QP_weight_f_Cb_B, 1, "Contact force magnitude penalty weight");
DEFINE_double(QP_mu, 1.0, "QP mu");  /* MBP defaults to mu1 == mu2 == 1.0 */
// TODO(rcory) Pass in QP_mu to brick and fingertip-sphere collision geoms.

// Define which fingers are used for the brick rotation.
DEFINE_bool(use_finger1, true, "Use finger1?");
DEFINE_bool(use_finger2, true, "Use finger2?");
DEFINE_bool(use_finger3, true, "Use finger3?");

DEFINE_bool(assume_zero_brick_damping, false,
            "Override brick joint damping with zero.");

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
  if (FLAGS_use_finger1) {
    finger_face_assignments.emplace(
        Finger::kFinger1,
        std::make_pair(BrickFace::kNegY, Eigen::Vector2d(-0.05, FLAGS_zc)));
  }
  if (FLAGS_use_finger2) {
    finger_face_assignments.emplace(
        Finger::kFinger2,
        std::make_pair(BrickFace::kPosY, Eigen::Vector2d(0.05, FLAGS_zc)));

  }
  if (FLAGS_use_finger3) {
    finger_face_assignments.emplace(
        Finger::kFinger3,
        std::make_pair(BrickFace::kNegZ, Eigen::Vector2d(FLAGS_yc, -0.05)));
  }
  return finger_face_assignments;
}

void GetQPPlannerOptions(const PlanarGripper& planar_gripper,
                         QPControlOptions* qpoptions) {
  double brick_damping = 0;
  if (!FLAGS_assume_zero_brick_damping) {
    brick_damping = planar_gripper.GetBrickDamping();
  }
  // Get the brick's Ixx moment of inertia (i.e., around the pinned axis).
  const int kIxx_index = 0;
  double brick_inertia = planar_gripper.GetBrickMoments()(kIxx_index);

  qpoptions->T_ = FLAGS_T;
  qpoptions->plan_dt = FLAGS_QP_plan_dt;
  qpoptions->theta0_ = FLAGS_theta0;
  qpoptions->thetaf_ = FLAGS_thetaf;
  qpoptions->QP_Kp_ro_ = FLAGS_QP_Kp_ro;
  qpoptions->QP_Kd_ro_ = FLAGS_QP_Kd_ro;
  qpoptions->QP_weight_thetaddot_error_ = FLAGS_QP_weight_thetaddot_error;
  qpoptions->QP_weight_f_Cb_B_ = FLAGS_QP_weight_f_Cb_B;
  qpoptions->QP_mu_ = FLAGS_QP_mu;
  qpoptions->brick_only_ = FLAGS_brick_only;
  qpoptions->viz_force_scale_ = FLAGS_viz_force_scale;
  qpoptions->brick_damping_ = brick_damping;
  qpoptions->brick_inertia_ = brick_inertia;
  qpoptions->brick_type_ = BrickType::PinBrick;
  qpoptions->finger_face_assignments_ = GetFingerFaceAssignments();
}

int DoMain() {
//  lcm::DrakeLcm lcm;
  systems::DiagramBuilder<double> builder;
  systems::lcm::LcmInterfaceSystem* lcm =
      builder.AddSystem<systems::lcm::LcmInterfaceSystem>();

  PlanarGripper planar_gripper;
  planar_gripper.SetupPinBrick("vertical");
  planar_gripper.zero_gravity(true);
  planar_gripper.Finalize();


  // Create a std::map to hold all input/output ports.
  std::map<std::string, const OutputPort<double>&> out_ports;
  std::map<std::string, const InputPort<double>&> in_ports;

  QPControlOptions qpoptions;
  GetQPPlannerOptions(planar_gripper, &qpoptions);

  // Add the QP controller to the diagram.
  const MultibodyPlant<double>& plant = planar_gripper.get_multibody_plant();
  AddGripperQPControllerToDiagram(plant, &builder, qpoptions, &in_ports,
                                  &out_ports);
  const auto lcm_sim = builder.AddSystem<PlanarGripperSimulationLcm>(
      plant.num_multibody_states(), lcm, kGripperLcmPeriod);

  // Connect the LCM sim outputs to the QP controller inputs.
  builder.Connect(lcm_sim->GetOutputPort("qp_estimated_plant_state"),
                  in_ports.at("qp_estimated_plant_state"));
  builder.Connect(lcm_sim->GetOutputPort("qp_finger_face_assignments"),
                  in_ports.at("qp_finger_face_assignments"));
  builder.Connect(lcm_sim->GetOutputPort("qp_desired_brick_accel"),
                  in_ports.at("qp_desired_brick_accel"));
  builder.Connect(lcm_sim->GetOutputPort("qp_desired_brick_state"),
                  in_ports.at("qp_desired_brick_state"));

  // Connect the QP controller outputs to the LCM sim inputs.
  builder.Connect(out_ports.at("qp_fingers_control"),
                  lcm_sim->GetInputPort("qp_fingers_control"));
  builder.Connect(out_ports.at("qp_brick_control"),
                  lcm_sim->GetInputPort("qp_brick_control"));

  auto owned_diagram = builder.Build();
  const systems::Diagram<double>* diagram = owned_diagram.get();
  systems::Simulator<double> simulator(std::move(owned_diagram));
  systems::Context<double>& diagram_context = simulator.get_mutable_context();

  // Make sure we receive one of each message before we begin.
  auto wait_for_new_message = [lcm](const auto& lcm_sub) {
    const int orig_count = lcm_sub.GetInternalMessageCount();
    LcmHandleSubscriptionsUntil(
        lcm, [&]() { return lcm_sub.GetInternalMessageCount() > orig_count; },
        10 /* timeout_millis */);
  };

  drake::log()->info("Waiting for initial messages...");
  wait_for_new_message(lcm_sim->get_estimated_plant_state_sub());
  wait_for_new_message(lcm_sim->get_finger_face_assignments_sub());
  wait_for_new_message(lcm_sim->get_brick_desired_sub());
  drake::log()->info("Received!");

  // Force a diagram update, to receive the first messages.
  systems::State<double>& diagram_state = diagram_context.get_mutable_state();
  diagram->CalcUnrestrictedUpdate(diagram_context, &diagram_state);

  // Get the first message and read it's time.
  const systems::Context<double>& est_plant_state_sub_context =
      diagram->GetSubsystemContext(
          lcm_sim->get_estimated_plant_state_sub(), diagram_context);
  auto first_msg = lcm_sim->get_estimated_plant_state_sub()
      .get_output_port()
      .Eval<lcmt_planar_plant_state>(est_plant_state_sub_context);

  const double t0 = first_msg.utime * 1e-6;
  diagram_context.SetTime(t0);

  // Send out the first message.
  diagram->Publish(diagram_context);

  drake::log()->info("Running controller...");
  simulator.Initialize();
  while (true) {
    wait_for_new_message(lcm_sim->get_estimated_plant_state_sub());
    wait_for_new_message(lcm_sim->get_finger_face_assignments_sub());
    wait_for_new_message(lcm_sim->get_brick_desired_sub());

    // Grab the next message.
    auto next_msg = lcm_sim->get_estimated_plant_state_sub()
        .get_output_port()
        .Eval<lcmt_planar_plant_state>(est_plant_state_sub_context);

    simulator.AdvanceTo(next_msg.utime * 1e-6);
    diagram->Publish(diagram_context);
  }
  // We should never reach here.
  return EXIT_FAILURE;
}

}  // namespace
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::planar_gripper::DoMain();
}
