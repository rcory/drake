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
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/constant_value_source.h"
#include "drake/systems/primitives/sine.h"
#include "drake/systems/primitives/multiplexer.h"
#include "drake/systems/primitives/adder.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/systems/primitives/zero_order_hold.h"
#include "drake/systems/primitives/demultiplexer.h"

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
DEFINE_double(simulation_time, 1.7,
              "Desired duration of the simulation in seconds.");
DEFINE_double(time_step, 1e-4,
            "If greater than zero, the plant is modeled as a system with "
            "discrete updates and period equal to this time_step. "
            "If 0, the plant is modeled as a continuous system.");
DEFINE_double(brick_z, 0, "Location of the brock on z-axis");
DEFINE_double(fix_input, false, "Fix the actuation inputs to zero?");
DEFINE_double(penetration_allowance, 0.1, "Penetration allowance.");
DEFINE_double(stiction_tolerance, 1e-3, "MBP v_stiction_tolerance");
DEFINE_double(j1, -0.63, "j1");
DEFINE_double(j2, 1.4, "j2");

DEFINE_double(fy, .1, "fy");
DEFINE_double(fz, -0.3, "fz");
DEFINE_double(Kd, 0.05, "joint damping Kd");
DEFINE_double(kpz, 0, "z-axis position gain");
DEFINE_double(kdz, 0, "z-axis derivative gain");
DEFINE_double(kfy, 20, "y-axis force gain");
DEFINE_double(kfz, 25, "z-axis force gain");
DEFINE_double(finger_x_offset, 0,
              "x-coordinate offset for welding the finger base.");
DEFINE_double(brick_theta0, 0, "initial brick angle");
DEFINE_double(K_compliance, 20, "Impedance control stiffness.");
DEFINE_double(D_damping, 1.0, "Impedance control damping.");
DEFINE_bool(force_direct_control, false, "Force direct force control?");

template<typename T>
void WeldFingerFrame(multibody::MultibodyPlant<T> *plant) {
  // This function is copied and adapted from planar_gripper_simulation.py
  const double outer_radius = 0.19;
  const double f1_angle = 0;
  const math::RigidTransformd XT(math::RollPitchYaw<double>(0, 0, 0),
                                 Eigen::Vector3d(0, 0, outer_radius));

  // Weld the first finger.
  Eigen::Vector3d p_offset(FLAGS_finger_x_offset, 0, 0);
  math::RigidTransformd X_PC1(math::RollPitchYaw<double>(f1_angle, 0, 0),
                              p_offset);
  X_PC1 = X_PC1 * XT;
  const multibody::Frame<T> &finger1_base_frame =
      plant->GetFrameByName("finger_base");
  plant->WeldFrames(plant->world_frame(), finger1_base_frame, X_PC1);
}

// Force controller with pure gravity compensation (no dynamics compensation
// yet). Regulates position in z, and force in y.
class ForceController : public systems::LeafSystem<double> {
 public:
  ForceController(MultibodyPlant<double>& plant)
      : plant_(plant), finger_index_(plant.GetModelInstanceByName("finger")) {
    // Make context with default parameters.
    plant_context_ = plant.CreateDefaultContext();

    force_desired_input_port_ =
        this->DeclareVectorInputPort(
                "f_d", systems::BasicVector<double>(2 /* num forces */))
            .get_index();
    const int kNumFingerVelocities = 2;
    state_actual_input_port_ =  // actual state of the finger
        this->DeclareVectorInputPort(
                "xa", systems::BasicVector<double>(2 * kNumFingerVelocities))
            .get_index();
    // desired state of the fingertip (y, z, ydot, zdot)
    const int kNumTipVelocities = 2;
    tip_state_desired_input_port_ =
        this->DeclareVectorInputPort(
                "xd", systems::BasicVector<double>(2 * kNumTipVelocities))
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
    output_calc.setZero();
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
    // assume only zero or one contact is possible.
    if (contact_results.num_point_pair_contacts() > 0) {
      auto contact_info = contact_results.point_pair_contact_info(0);
      force_sim = contact_info.contact_force();
    }
    // Keep only the last two components of the force. Negative because this
    // force returns as the force felt by the fingertip.
    Eigen::Vector2d force_act = -force_sim.tail<2>();

    // Set the plant's position and velocity within the context.
    plant_.SetPositionsAndVelocities(plant_context_.get(), finger_index_,
                                     state);
    Eigen::Vector2d torque_calc(0, 0);

    // Gravity compensation.
    // TODO(rcory) why is this not the first two components of
    //  CalcGravityGeneralizedForces? Joint ordering indicates this should be
    //  the case...
    torque_calc =
        -plant_.CalcGravityGeneralizedForces(*plant_context_).segment<2>(1);

    // Compute the Jacobian.
    // For the finger/1-dof brick case, the plant consists of 3 dofs total (2 of
    // which belong to the finger). The resultant Jacobian will be of size 6 x 3.
    Eigen::Matrix<double, 6, 3> Jv_V_WFtip(6, 3);
    const multibody::Frame<double>& l2_frame =
        plant_.GetBodyByName("finger_link2").body_frame();
    const multibody::Frame<double>& base_frame =
        plant_.GetBodyByName("finger_base").body_frame();
    const Vector3<double> p_L2Ftip(0, 0, -0.086);
    plant_.CalcJacobianSpatialVelocity(
        *plant_context_, multibody::JacobianWrtVariable::kV, l2_frame, p_L2Ftip,
        base_frame, base_frame, &Jv_V_WFtip);

    // Extract the planar translational part of the Jacobian.
    // The last two rows of Jv_V_WFtip correspond to y-z.
    // TODO(rcory) for some odd reason, the last two columns of the Jacobian
    // correspond to j1 and j2 for the brick....odd.
    Eigen::Matrix<double, 2, 2> J(2, 2);
    J.block<1, 2>(0, 0) = Jv_V_WFtip.block<1, 2>(4, 1);  // should be (4,0) and (5,0)?
    J.block<1, 2>(1, 0) = Jv_V_WFtip.block<1, 2>(5, 1);

    // Get the fingertip position
    const multibody::Frame<double>& L2_frame =
        plant_.GetBodyByName("finger_link2").body_frame();
    Eigen::Vector3d p_WFtip(0, 0, 0);
    plant_.CalcPointsPositions(*plant_context_, L2_frame, p_L2Ftip,
                               plant_.world_frame(), &p_WFtip);

    // drake::log()->info("p_tip: \n{}", p_WFtip);                           

    // Force control gains.
    Eigen::Matrix<double, 2, 2> Kf(2,2);
    Kf << FLAGS_kfy, 0, 0, FLAGS_kfz;  // Allow force control in both y and z

    // Regulate force in y (in world frame)
    auto delta_f = force_des - force_act;
    // auto fy_command = Kf * delta_f + force_des;  //TODO(rcory) bug?
    auto fy_command = Kf * delta_f;
    // auto fy_command = Kf Δf + Ki ∫ Δf dt - Kp p_e + f_d  // More general.

    // Regulate position in z (in world frame)
    auto tip_velocity_actual = J * state.segment<2>(2);  // does MBP provide this?
    auto delta_pos = tip_state_desired.head<2>() - p_WFtip.tail<2>();
    auto delta_vel =
        tip_state_desired.tail<2>() - tip_velocity_actual.head<2>();
    Eigen::Matrix<double, 2, 2> Kp_pos(2,2), Kd_pos(2,2);
    Kp_pos << 0, 0, 0, FLAGS_kpz;  // position control only in z
    Kd_pos << 0, 0, 0, FLAGS_kdz;
    auto fz_command = Kp_pos * delta_pos + Kd_pos * delta_vel;

    if (FLAGS_force_direct_control || contact_results.num_point_pair_contacts() > 0) {
      drake::log()->info("contact > 0");
      // Adds Joint damping.
      Eigen::Matrix2d Kd;
      Kd << FLAGS_Kd, 0, 0, FLAGS_Kd;
      torque_calc += -Kd * state.segment<2>(2);

      // Torque due to hybrid position/force control
      torque_calc += J.transpose() * (force_des + fy_command + fz_command);
    } else {  // regulate the fingertip position back to the surface w/ impedance control
      drake::log()->info("contact == 0!!");
      // implement a simple spring law for now (-kx), i.e., just compliance control
      const double target_z_position = 0.1;  // arbitrary
      const double K = FLAGS_K_compliance;  // spring const
      const double D = FLAGS_D_damping;
      double z_force_desired = -K * (target_z_position - p_WFtip(2)) - D * tip_velocity_actual(1);
      Vector2<double> imp_force_desired(0, z_force_desired);
      torque_calc += J.transpose() * imp_force_desired;
    }

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
  ModelInstanceIndex finger_index_{};
};

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
      Parser(&plant, &scene_graph).AddModelFromFile(full_name, "finger");
  WeldFingerFrame<double>(&plant);

//   Adds the object to be manipulated.
 auto object_file_name =
     FindResourceOrThrow("drake/examples/planar_gripper/1dof_brick.sdf");
 Parser(&plant).AddModelFromFile(object_file_name, "object");

  // Add gravity
  Vector3<double> gravity(0, 0, -9.81);
  plant.mutable_gravity_field().set_gravity_vector(gravity);
//  control_plant.mutable_gravity_field().set_gravity_vector(gravity);

  // Now the model is complete.
  plant.Finalize();
//  control_plant.Finalize();

  // Set the penetration allowance for the simulation plant only
  plant.set_penetration_allowance(FLAGS_penetration_allowance);
  plant.set_stiction_tolerance(FLAGS_stiction_tolerance);

  // Sanity check on the availability of the optional source id before using it.
  DRAKE_DEMAND(plant.geometry_source_is_registered());

  // Connect the force controler
  auto zoh = builder.AddSystem<systems::ZeroOrderHold<double>>(
  1e-3, Value<ContactResults<double>>());
  Vector2<double> constfv(FLAGS_fy, FLAGS_fz);  // {fy, fz}
  auto force_controller = builder.AddSystem<ForceController>(plant);
  auto const_force_src = builder.AddSystem<systems::ConstantVectorSource>(constfv);
  builder.Connect(const_force_src->get_output_port(),
                  force_controller->get_force_desired_input_port());

  builder.Connect(plant.get_state_output_port(finger_index),
                  force_controller->get_state_actual_input_port());
  builder.Connect(plant.get_contact_results_output_port(),
                  zoh->get_input_port());
  builder.Connect(zoh->get_output_port(),
                  force_controller->get_contact_results_input_port());

  // aux debugging info
  std::vector<int> sizes = {2, 2, 1}; // tau_des, f_des, ytip
  auto demux = builder.AddSystem<systems::Demultiplexer<double>>(sizes);
  builder.Connect(force_controller->get_torque_output_port(),
                  demux->get_input_port(0));
  builder.Connect(demux->get_output_port(0),
                  plant.get_actuation_input_port(finger_index));

  // size 4 because we take in {y, ydot, z, zdot}. The position gain on y will
  // be zero (since y is force controlled).
  Vector4<double> constpv;  // regulate z position to the starting tip position
  constpv.setZero();
  constpv << -0.0248002 , 0.0590716;
  auto const_pos_src = builder.AddSystem<systems::ConstantVectorSource>(constpv);
  builder.Connect(const_pos_src->get_output_port(),
                  force_controller->get_state_desired_input_port());

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
        plant.get_actuation_input_port().get_index(), tau_d);
  }

  // Set finger initial conditions.
  VectorX<double> finger_initial_conditions = VectorX<double>::Zero(4);
  finger_initial_conditions << FLAGS_j1, FLAGS_j2, 0, 0;
  const RevoluteJoint<double>& sh_pin1 =
      plant.GetJointByName<RevoluteJoint>("finger_ShoulderJoint");
  const RevoluteJoint<double>& el_pin1 =
      plant.GetJointByName<RevoluteJoint>("finger_ElbowJoint");
  sh_pin1.set_angle(&plant_context, finger_initial_conditions(0));
  el_pin1.set_angle(&plant_context, finger_initial_conditions(1));

 // Set the brick's initial condition.
 const RevoluteJoint<double>& brick_pin =
     plant.GetJointByName<RevoluteJoint>("brick_pin_joint");
 brick_pin.set_angle(&plant_context, FLAGS_brick_theta0);

  PrintJointOrdering(plant);

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
