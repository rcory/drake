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
DEFINE_double(simulation_time, 5.0,
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
DEFINE_double(fz, -10, "Desired end effector force");
DEFINE_double(Kd, 0.3, "joint damping Kd");
DEFINE_double(kpy, 1200.0*0, "kpy");

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
    state_input_port_ =
        this->DeclareVectorInputPort(
                "x", systems::BasicVector<double>(2 * plant.num_velocities()))
            .get_index();
    contact_results_input_port_ =
        this->DeclareAbstractInputPort("contact_results",
                                       Value<ContactResults<double>>{})
            .get_index();
    torque_output_port_ =
        this->DeclareVectorOutputPort(
                "tau", systems::BasicVector<double>(plant.num_actuators()),
                &ForceController::CalcTauOutput)
            .get_index();
  }

  const InputPort<double>& get_force_desired_input_port() const {
      return this->get_input_port(force_desired_input_port_);
  }

  const InputPort<double>& get_state_input_port() const {
      return this->get_input_port(state_input_port_);
  }

  const InputPort<double>& get_contact_results_input_port() const {
      return this->get_input_port(contact_results_input_port_);
  }

  const OutputPort<double>& get_torque_output_port() const {
      return this->get_output_port(torque_output_port_);
  }

  void CalcTauOutput(const systems::Context<double>& context,
                  systems::BasicVector<double>* output_vector) const {
    auto torque_calc = output_vector->get_mutable_value();
    auto force_des =
        this->EvalVectorInput(context, force_desired_input_port_)->get_value();
    auto state =
        this->EvalVectorInput(context, state_input_port_)->get_value();

   const auto& contact_results =
        get_contact_results_input_port().Eval<ContactResults<double>>(context);
    unused(contact_results);

    // Set the plant's position and velocity within the context.
    plant_.SetPositionsAndVelocities(plant_context_.get(), state);
    torque_calc.setZero();

    // Gravity compensation.
    torque_calc =
        -plant_.CalcGravityGeneralizedForces(*plant_context_);

    // Damping.
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

    // Extract the planar Jacobian.
    Eigen::Matrix<double, 3, 2> J(3, 2);
    J.block<1, 2>(0, 0) = Jv_V_WFtip.block<1, 2>(4, 0);
    J.block<1, 2>(1, 0) = Jv_V_WFtip.block<1, 2>(5, 0);
    J.block<1, 2>(2, 0) = Jv_V_WFtip.block<1, 2>(0, 0);

    // Get the fingertip position
    const multibody::Frame<double>& L2_frame =
        plant_.GetBodyByName("finger_link2").body_frame();
    Eigen::Vector3d p_WFtip(0, 0, 0);
    plant_.CalcPointsPositions(*plant_context_, L2_frame, p_L2Ftip,
                               plant_.world_frame(), &p_WFtip);

    // Add the desired end effector force.
    Eigen::Vector2d y_command(0, 0);
    y_command(0) = FLAGS_kpy * (0.04 - p_WFtip(1));
    torque_calc += J.transpose() * (force_des + y_command);
  }

 private:
  MultibodyPlant<double>& plant_;
  // This context is used solely for setting generalized positions and
  // velocities in plant_.
  std::unique_ptr<systems::Context<double>> plant_context_;  
  InputPortIndex force_desired_input_port_{};
  InputPortIndex state_input_port_{};
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

  // Sanity check on the availability of the optional source id before using it.
  DRAKE_DEMAND(plant.geometry_source_is_registered());
  unused(plant_id);

  // Connect the force controler
  auto zoh = builder.AddSystem<systems::ZeroOrderHold<double>>(
      1e-3, Value<ContactResults<double>>());  
  Vector2<double> constv(0, FLAGS_fz);
  auto force_controller = builder.AddSystem<ForceController>(plant);
  auto const_src = builder.AddSystem<systems::ConstantVectorSource>(constv);
  builder.Connect(const_src->get_output_port(),
                  force_controller->get_force_desired_input_port());

  builder.Connect(plant.get_state_output_port(),
                  force_controller->get_state_input_port());
  builder.Connect(plant.get_contact_results_output_port(),
                  zoh->get_input_port());
  builder.Connect(zoh->get_output_port(),
                  force_controller->get_contact_results_input_port());
  builder.Connect(force_controller->get_torque_output_port(),
                  plant.get_actuation_input_port());

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
        plant.get_applied_generalized_force_input_port().get_index(), tau_d);
  }

  // Set initial conditions.
  VectorX<double> finger_initial_conditions = VectorX<double>::Zero(4);
//  finger_initial_conditions << -0.65, 1.7, 0, 0;
  finger_initial_conditions << -0.681, 1.066, 0, 0;

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
