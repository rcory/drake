#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/common/text_logging_gflags.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/zero_order_hold.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/systems/primitives/demultiplexer.h"
#include "drake/systems/primitives/multiplexer.h"
#include "drake/systems/primitives/sine.h"
#include "drake/systems/lcm/connect_lcm_scope.h"

namespace drake {
namespace examples {
namespace planar_gripper {
namespace {

using geometry::SceneGraph;

using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::multibody::RevoluteJoint;
using drake::multibody::PrismaticJoint;
using drake::math::RigidTransform;
using drake::math::RollPitchYaw;
using drake::multibody::JointActuatorIndex;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::ConnectContactResultsToDrakeVisualizer;
using lcm::DrakeLcm;
using drake::multibody::ContactResults;
using systems::InputPortIndex;
using systems::OutputPortIndex;
using systems::OutputPort;
using systems::InputPort;

DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_double(simulation_time, 8.0,
              "Desired duration of the simulation in seconds.");
DEFINE_double(time_step, 1e-4,
            "If greater than zero, the plant is modeled as a system with "
            "discrete updates and period equal to this time_step. "
            "If 0, the plant is modeled as a continuous system.");
DEFINE_bool(fix_input, false, "Fix the actuation inputs to zero?");
DEFINE_bool(use_brick, false,
            "True if sim should use the 1dof brick (revolute), false if it "
            "should use the 1dof surface.");
DEFINE_double(penetration_allowance, 0.005, "Penetration allowance.");
DEFINE_double(stiction_tolerance, 1e-3, "MBP v_stiction_tolerance");
DEFINE_double(fz, -10.0, "Desired end effector force");
DEFINE_double(Kd, 0.3, "joint damping Kd");

template<typename T>
void WeldFingerFrame(multibody::MultibodyPlant<T> *plant) {
  // This function is copied and adapted from planar_gripper_simulation.py
  const double outer_radius = 0.19;
  const double f1_angle = 0;
  const math::RigidTransformd XT(math::RollPitchYaw<double>(0, 0, 0),
                                 Eigen::Vector3d(0, 0, outer_radius));

  // Weld the first finger.
  math::RigidTransformd X_PC1(math::RollPitchYaw<double>(f1_angle, 0, 0),
                              Eigen::Vector3d::Zero());
  X_PC1 = X_PC1 * XT;
  const multibody::Frame<T> &finger1_base_frame =
      plant->GetFrameByName("finger_base");
  plant->WeldFrames(plant->world_frame(), finger1_base_frame, X_PC1);
}

// Force controller with pure gravity compensation (no dynamics compensation
// yet). Regulates position in y, and force in z.
class ForceController : public systems::LeafSystem<double> {
 public:
  ForceController(MultibodyPlant<double>& plant)
      : plant_(plant) {
    // Make context with default parameters.
    plant_context_ = plant.CreateDefaultContext();

    force_desired_input_port_ =
        this->DeclareVectorInputPort(
                "f_d", systems::BasicVector<double>(2 /* num forces */))
            .get_index();
    state_actual_input_port_ =  // actual state
        this->DeclareVectorInputPort(
                "xa", systems::BasicVector<double>(2 * plant.num_velocities()))
            .get_index();
    // desired state of the fingertip (y, z, ydot, zdot)
    tip_state_desired_input_port_ =
        this->DeclareVectorInputPort(
                "xd", systems::BasicVector<double>(2 * plant.num_velocities()))
            .get_index();
    contact_results_input_port_ =
        this->DeclareAbstractInputPort("contact_results",
                                       Value<ContactResults<double>>{})
            .get_index();

    const int kDebuggingOutputs = 3;
    torque_output_port_ =
        this->DeclareVectorOutputPort(
                "tau",
                systems::BasicVector<double>(plant.num_actuators() +
                                             kDebuggingOutputs),
                &ForceController::CalcTauOutput)
            .get_index();
  }

  const InputPort<double>& get_force_desired_input_port() const {
      return this->get_input_port(force_desired_input_port_);
  }

  const InputPort<double>& get_state_actual_input_port() const {
      return this->get_input_port(state_actual_input_port_);
  }

  const InputPort<double>& get_state_desired_input_port() const {
    return this->get_input_port(tip_state_desired_input_port_);
  }

  const InputPort<double>& get_contact_results_input_port() const {
      return this->get_input_port(contact_results_input_port_);
  }

  const OutputPort<double>& get_torque_output_port() const {
      return this->get_output_port(torque_output_port_);
  }

  void CalcTauOutput(const systems::Context<double>& context,
                  systems::BasicVector<double>* output_vector) const {
    auto output_calc = output_vector->get_mutable_value();
    auto force_des =
        this->EvalVectorInput(context, force_desired_input_port_)->get_value();
    auto state =
        this->EvalVectorInput(context, state_actual_input_port_)->get_value();
    auto tip_state_desired =
        this->EvalVectorInput(context, tip_state_desired_input_port_)->get_value();

    // Get the actual contact force.
    const auto& contact_results =
        get_contact_results_input_port().Eval<ContactResults<double>>(context);

    Eigen::Vector3d force_sim(0, 0, 0);
    if (contact_results.num_contacts() > 0) {
      auto contact_info = contact_results.point_pair_contact_info(0);
      force_sim = contact_info.contact_force();
    }
    // Keep only the last two components of the force. Negative because this
    // force returns as the force felt by the fingertip.
    Eigen::Vector2d force_act = -force_sim.tail<2>();

    // Set the plant's position and velocity within the context.
    plant_.SetPositionsAndVelocities(plant_context_.get(), state);
    Eigen::Vector2d torque_calc(0, 0);

    // Gravity compensation.
    torque_calc =
        -plant_.CalcGravityGeneralizedForces(*plant_context_);

    // Adds Joint damping.
    Eigen::Matrix2d Kd;
    Kd << FLAGS_Kd, 0, 0, FLAGS_Kd;
    torque_calc += -Kd * state.tail(2);

    // Compute the jacobian.
    Eigen::Matrix<double, 6, 2> Jv_V_WFtip(6, 2);
    const multibody::Frame<double>& l2_frame =
        plant_.GetBodyByName("finger_link2").body_frame();
    const multibody::Frame<double>& base_frame =
        plant_.GetBodyByName("finger_base").body_frame();        
    const Vector3<double> p_L2Ftip(0, 0, -0.086);
    plant_.CalcJacobianSpatialVelocity(
        *plant_context_, multibody::JacobianWrtVariable::kV, l2_frame, p_L2Ftip,
        base_frame, base_frame, &Jv_V_WFtip);

    // Extract the planar translational part of the Jacobian.
    Eigen::Matrix<double, 2, 2> J(2, 2);
    J.block<1, 2>(0, 0) = Jv_V_WFtip.block<1, 2>(4, 0);
    J.block<1, 2>(1, 0) = Jv_V_WFtip.block<1, 2>(5, 0);

    // Get the fingertip position
    const multibody::Frame<double>& L2_frame =
        plant_.GetBodyByName("finger_link2").body_frame();
    Eigen::Vector3d p_WFtip(0, 0, 0);
    plant_.CalcPointsPositions(*plant_context_, L2_frame, p_L2Ftip,
                               plant_.world_frame(), &p_WFtip);

    // Force control gains.
    Eigen::Matrix<double, 2, 2> Kf(2,2);
    Kf << 0, 0, 0, 10;  // Gain only on z component (y regulates position)

    // Regulate force in z (in world frame)
    auto delta_f = force_des - force_act;
    auto fz_command = Kf * delta_f + force_des;
    // auto fz_command = Kf Δf + Ki ∫ Δf dt - Kp p_e + f_d  // More general.

    // Regulate position in y (in world frame)
    auto tip_velocity_actual = J * state;  // does MBP provide this?
    auto delta_pos = tip_state_desired.head<2>() - p_WFtip.tail<2>();
    auto delta_vel =
        tip_state_desired.tail<2>() - tip_velocity_actual.head<2>();
    Eigen::Matrix<double, 2, 2> Kp_pos(2,2), Kd_pos(2,2);
    Kp_pos << 5e3, 0, 0, 0;  // position control only in y
    Kd_pos << 0, 0, 0, 0;  // already lots of damping
    auto fy_command = Kp_pos * delta_pos + Kd_pos * delta_vel;

    // Torque due to hybrid position/force control
    torque_calc += J.transpose() * (force_des + fz_command + fy_command);

    // The output for calculated torques.
    output_calc.head<2>() = torque_calc;

    // These are just auxiliary debugging outputs.
    output_calc.segment<2>(2) = force_des + fz_command + fy_command;
    output_calc(4) = delta_pos(0);
  }

 private:
  MultibodyPlant<double>& plant_;
  // This context is used solely for setting generalized positions and
  // velocities in plant_.
  std::unique_ptr<systems::Context<double>> plant_context_;  
  InputPortIndex force_desired_input_port_{};
  InputPortIndex state_actual_input_port_{};
  InputPortIndex tip_state_desired_input_port_{};
  InputPortIndex contact_results_input_port_{};
  OutputPortIndex torque_output_port_{};
};

int do_main() {
  systems::DiagramBuilder<double> builder;

  SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
  scene_graph.set_name("scene_graph");

  // Make and add the planar_gripper model.
  const std::string full_name = FindResourceOrThrow(
      "drake/examples/planar_gripper/planar_finger.sdf");
  MultibodyPlant<double>& plant =
      *builder.AddSystem<MultibodyPlant>(FLAGS_time_step);
  auto plant_id = Parser(&plant, &scene_graph).AddModelFromFile(full_name);
  WeldFingerFrame<double>(&plant);

  // Adds the object to be manipulated.
  if (FLAGS_use_brick) {
    auto object_file_name =
        FindResourceOrThrow("drake/examples/planar_gripper/1dof_brick.sdf");
    Parser(&plant).AddModelFromFile(object_file_name, "object");
  } else {
    auto object_file_name =
        FindResourceOrThrow("drake/examples/planar_gripper/fixed_surface.sdf");
    Parser(&plant).AddModelFromFile(object_file_name, "object");
  }

  // Create the controlled plant. Contains only the fingers (no objects).
  MultibodyPlant<double> control_plant(FLAGS_time_step);
  Parser(&control_plant).AddModelFromFile(full_name);
  WeldFingerFrame<double>(&control_plant);

  // Add gravity
  Vector3<double> gravity(0, 0, -9.81);
  plant.mutable_gravity_field().set_gravity_vector(gravity);
  control_plant.mutable_gravity_field().set_gravity_vector(gravity);

  // Now the model is complete.
  plant.Finalize();
  control_plant.Finalize();

  // Set the penetration allowance for the simulation plant only
  plant.set_penetration_allowance(FLAGS_penetration_allowance);
  plant.set_stiction_tolerance(FLAGS_stiction_tolerance);

  // Sanity check on the availability of the optional source id before using it.
  DRAKE_DEMAND(plant.geometry_source_is_registered());
  unused(plant_id);

  // Connect the force controler
  auto zoh = builder.AddSystem<systems::ZeroOrderHold<double>>(
      1e-3, Value<ContactResults<double>>());  
  Vector2<double> constfv(0, FLAGS_fz);
  auto force_controller = builder.AddSystem<ForceController>(plant);
  auto const_force_src = builder.AddSystem<systems::ConstantVectorSource>(constfv);
  builder.Connect(const_force_src->get_output_port(),
                  force_controller->get_force_desired_input_port());

  builder.Connect(plant.get_state_output_port(),
                  force_controller->get_state_actual_input_port());
  builder.Connect(plant.get_contact_results_output_port(),
                  zoh->get_input_port());
  builder.Connect(zoh->get_output_port(),
                  force_controller->get_contact_results_input_port());

  std::vector<int> sizes = {2, 2, 1}; // tau_des, f_des, ytip
  auto demux = builder.AddSystem<systems::Demultiplexer<double>>(sizes);
  builder.Connect(force_controller->get_torque_output_port(),
                  demux->get_input_port(0));
  builder.Connect(demux->get_output_port(0), plant.get_actuation_input_port());

  // Connect a sine wave reference trajectory (for y position control)
  auto sine_src =
      builder.AddSystem<systems::Sine<double>>(.04, 3.0, 3 * M_PI / 2.0, 1);
  auto mux = builder.AddSystem<systems::Multiplexer>(4);
  builder.Connect(sine_src->get_output_port(0), mux->get_input_port(0));
  builder.Connect(sine_src->get_output_port(1), mux->get_input_port(2));
  builder.Connect(mux->get_output_port(0),
                  force_controller->get_state_desired_input_port());
  auto zero_scalar_src =
      builder.AddSystem<systems::ConstantVectorSource>(Vector1d::Zero());
  builder.Connect(zero_scalar_src->get_output_port(), mux->get_input_port(1));
  builder.Connect(zero_scalar_src->get_output_port(), mux->get_input_port(3));

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

  // ====== Publish the desired force calculation. =========
  systems::lcm::ConnectLcmScope(demux->get_output_port(0), "TORQUE_COMMAND",
                                &builder, &lcm);
  systems::lcm::ConnectLcmScope(demux->get_output_port(1), "FORCE_COMMAND",
                                &builder, &lcm);
  systems::lcm::ConnectLcmScope(demux->get_output_port(2), "ERROR_Y",
                                &builder, &lcm);

  auto diagram = builder.Build();

  // Create a context for this system:
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  if (FLAGS_fix_input) {
    VectorX<double> tau_d(2);
    tau_d << 0, 0;
    plant_context.FixInputPort(
        plant.get_applied_generalized_force_input_port().get_index(), tau_d);
  }

  // Set initial conditions.
  VectorX<double> finger_initial_conditions = VectorX<double>::Zero(4);
//  finger_initial_conditions << -0.65, 1.7, 0, 0;
//  finger_initial_conditions << -0.681, 1.066, 0, 0;
  finger_initial_conditions << -0.8112, 0.8667, 0, 0;

  // Finger 1
  const RevoluteJoint<double>& sh_pin1 =
      plant.GetJointByName<RevoluteJoint>("finger_ShoulderJoint");
  const RevoluteJoint<double>& el_pin1 =
      plant.GetJointByName<RevoluteJoint>("finger_ElbowJoint");
  sh_pin1.set_angle(&plant_context, finger_initial_conditions(0));
  el_pin1.set_angle(&plant_context, finger_initial_conditions(1));
  sh_pin1.set_angular_rate(&plant_context, finger_initial_conditions(2));
  sh_pin1.set_angular_rate(&plant_context, finger_initial_conditions(3));

  // Set the brick's initial condition.
  if (FLAGS_use_brick) {
    const RevoluteJoint<double> &box_pin =
        plant.GetJointByName<RevoluteJoint>("box_pin_joint");
    box_pin.set_angle(&plant_context, 0);
  }

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
  drake::logging::HandleSpdlogGflags();
  return drake::examples::planar_gripper::do_main();
}