/// @file
///
/// Implements an UDP based finger/brick QP controller.

#include <gflags/gflags.h>

#include "drake/examples/planar_gripper/finger_brick_control.h"
#include "drake/examples/planar_gripper/planar_gripper.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/examples/planar_gripper/planar_gripper_udp.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace examples {
namespace planar_gripper {
namespace {

// TODO(rcory) Move all common flags to a shared YAML file.
DEFINE_double(simulation_time, 3.0, "Amount of time to simulate.");
DEFINE_double(viz_force_scale, 1,
              "scale factor for visualizing spatial force arrow");
DEFINE_bool(brick_only, false, "Only simulate brick (no finger).");

// UDP parameters
DEFINE_int32(publisher_local_port, 1101,
             "local port number for UDP publisher.");
DEFINE_int32(publisher_remote_port, 1100,
             "remote port number for UDP publisher.");
// I convert the IP address of my computer to unsigned long through
// https://www.smartconversion.com/unit_conversion/IP_Address_Converter.aspx
DEFINE_uint64(publisher_remote_address, 0,
              "remote IP address for UDP publisher.");
DEFINE_int32(receiver_local_port, 1102, "local port number for UDP receiver.");

// QP task parameters
DEFINE_double(theta0, -M_PI_4 + 0.2, "initial theta (rad)");
DEFINE_double(thetaf, M_PI_4, "final theta (rad)");
DEFINE_double(T, 1.5, "time horizon (s)");
DEFINE_double(QP_plan_dt, 0.002, "The QP planner's timestep.");

DEFINE_double(QP_kp_r, 150, "QP controller rotational proportional gain");
DEFINE_double(QP_kd_r, 50, "QP controller rotational derivative gain");
DEFINE_double(QP_weight_thetaddot_error, 1, "thetaddot error weight.");
DEFINE_double(QP_weight_f_Cb_B, 1, "Contact force magnitude penalty weight");
DEFINE_double(QP_mu, 1.0, "QP mu"); /* MBP defaults to mu1 == mu2 == 1.0 */
// TODO(rcory) Pass in QP_mu to brick and fingertip-sphere collision geoms.

// Define which fingers are used for the brick rotation.
DEFINE_bool(use_finger1, true, "Use finger1?");
DEFINE_bool(use_finger2, true, "Use finger2?");
DEFINE_bool(use_finger3, true, "Use finger3?");

DEFINE_bool(assume_zero_brick_damping, false,
            "Override brick joint damping with zero.");

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
                         QPControlOptions* qpoptions) {
  double brick_damping = 0;
  if (!FLAGS_assume_zero_brick_damping) {
    brick_damping = planar_gripper.GetBrickPinJointDamping();
  }
  // Get the brick's Ixx moment of inertia (i.e., around the pinned axis).
  const int kIxx_index = 0;
  double brick_inertia = planar_gripper.GetBrickMoments()(kIxx_index);

  qpoptions->T_ = FLAGS_T;
  qpoptions->plan_dt = FLAGS_QP_plan_dt;
  qpoptions->brick_goal_.theta_goal = FLAGS_thetaf;
  qpoptions->QP_kp_r_ = FLAGS_QP_kp_r;
  qpoptions->QP_kd_r_ = FLAGS_QP_kd_r;
  qpoptions->QP_weight_thetaddot_error_ = FLAGS_QP_weight_thetaddot_error;
  qpoptions->QP_weight_f_Cb_B_ = FLAGS_QP_weight_f_Cb_B;
  qpoptions->QP_mu_ = FLAGS_QP_mu;
  qpoptions->brick_only_ = FLAGS_brick_only;
  qpoptions->viz_force_scale_ = FLAGS_viz_force_scale;
  qpoptions->brick_rotational_damping_ = brick_damping;
  qpoptions->brick_inertia_ = brick_inertia;
  qpoptions->brick_type_ = BrickType::PinBrick;
  qpoptions->brick_spatial_force_assignments_ =
      BrickSpatialForceAssignments(FingersToControl());
}

int DoMain() {
  //  lcm::DrakeLcm lcm;
  systems::DiagramBuilder<double> builder;

  PlanarGripper planar_gripper(0.0 /* time step */, ControlType::kTorque);
  planar_gripper.SetupPinBrick("vertical");
  planar_gripper.zero_gravity();
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
  const auto udp_sim = builder.AddSystem<PlanarGripperSimulationUDP>(
      plant.num_multibody_states(), kNumFingers,
      planar_gripper.get_num_brick_states(),
      planar_gripper.get_num_brick_velocities(), FLAGS_publisher_local_port,
      FLAGS_publisher_remote_port, FLAGS_publisher_remote_address,
      FLAGS_receiver_local_port, kGripperUdpStatusPeriod);

  // Connect the UDP sim outputs to the QP controller inputs.
  builder.Connect(udp_sim->GetOutputPort("qp_estimated_plant_state"),
                  in_ports.at("qp_estimated_plant_state"));
  builder.Connect(udp_sim->GetOutputPort("qp_finger_face_assignments"),
                  in_ports.at("qp_finger_face_assignments"));
  builder.Connect(udp_sim->GetOutputPort("qp_desired_brick_accel"),
                  in_ports.at("qp_desired_brick_accel"));
  builder.Connect(udp_sim->GetOutputPort("qp_desired_brick_state"),
                  in_ports.at("qp_desired_brick_state"));

  // Connect the QP controller outputs to the UDP sim inputs.
  builder.Connect(out_ports.at("qp_fingers_control"),
                  udp_sim->GetInputPort("qp_fingers_control"));
  builder.Connect(out_ports.at("qp_brick_control"),
                  udp_sim->GetInputPort("qp_brick_control"));

  auto owned_diagram = builder.Build();
  const systems::Diagram<double>* diagram = owned_diagram.get();
  systems::Simulator<double> simulator(std::move(owned_diagram));
  systems::Context<double>& diagram_context = simulator.get_mutable_context();

  diagram_context.SetTime(0);

  // Send out the first message.
  diagram->Publish(diagram_context);

  drake::log()->info("Running controller...");
  simulator.Initialize();
  simulator.AdvanceTo(10);
  return 0;
}

}  // namespace
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::planar_gripper::DoMain();
}
