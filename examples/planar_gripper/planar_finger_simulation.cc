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
#include "Eigen/src/Core/Matrix.h"
#include "drake/systems/primitives/sine.h"
#include "drake/systems/primitives/multiplexer.h"
#include "drake/systems/primitives/adder.h"
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

DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_double(simulation_time, 8.0,
              "Desired duration of the simulation in seconds.");
DEFINE_double(time_step, 1e-4,
            "If greater than zero, the plant is modeled as a system with "
            "discrete updates and period equal to this time_step. "
            "If 0, the plant is modeled as a continuous system.");
DEFINE_double(brick_z, 0, "Location of the brock on z-axis");
DEFINE_double(fix_input, false, "Fix the actuation inputs to zero?");
DEFINE_bool(use_brick, true,
            "True if sim should use the 1dof brick (revolute), false if it "
            "should use the 1dof surface.");

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

/// Converts the generalized force output of the ID controller (internally using
/// a control plant with only the finger) to the generalized force input for
/// the full simulation plant (containing finger and object).
class MakePlantGeneralizedForceArray : public systems::LeafSystem<double> {
 public:
  MakePlantGeneralizedForceArray(MultibodyPlant<double>& plant,
                                 ModelInstanceIndex gripper_instance)
      : plant_(plant), gripper_instance_(gripper_instance) {
    this->DeclareVectorInputPort(
        "input1", systems::BasicVector<double>(plant.num_actuators()));
    this->DeclareVectorOutputPort(
        "output1", systems::BasicVector<double>(plant.num_velocities()),
        &MakePlantGeneralizedForceArray::remap_output);
  }

  void remap_output(const systems::Context<double>& context,
                    systems::BasicVector<double>* output_vector) const {
    auto output_value = output_vector->get_mutable_value();
    auto input_value = this->EvalVectorInput(context, 0)->get_value();

    output_value.setZero();
    plant_.SetVelocitiesInArray(gripper_instance_, input_value, &output_value);
  }

 private:
  MultibodyPlant<double>& plant_;
  ModelInstanceIndex gripper_instance_;
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

  // Sanity check on the availability of the optional source id before using it.
  DRAKE_DEMAND(plant.geometry_source_is_registered());

  // Inverse Dynamics Source
  VectorX<double> Kp(2), Kd(2), Ki(2);
  Kp = VectorX<double>::Ones(2) * 1500;
  Ki = VectorX<double>::Ones(2) * 500;
  Kd = VectorX<double>::Ones(2) * 500;

  auto id_controller =
      builder.AddSystem<systems::controllers::InverseDynamicsController>(
          control_plant, Kp, Ki, Kd, false);

  // Connect the ID controller
  builder.Connect(plant.get_state_output_port(plant_id),
                  id_controller->get_input_port_estimated_state());

  // Sine reference.
  Eigen::Vector2d amplitudes(0.2, 0.7);
  Eigen::Vector2d frequencies(3, 6.1);
  Eigen::Vector2d phases(0, 0.2);
  auto sine_source =
      builder.AddSystem<systems::Sine<double>>(amplitudes, frequencies, phases);
  std::vector<int> mux_size{2, 2};
  auto smux = builder.AddSystem<systems::Multiplexer<double>>(mux_size);

  // Add Sine offsets.
  auto adder = builder.AddSystem<systems::Adder<double>>(2, 2);
  auto offsets = builder.AddSystem<systems::ConstantVectorSource<double>>(
      Eigen::Vector2d(-0.65, 1.21));
  builder.Connect(sine_source->get_output_port(0), adder->get_input_port(0));
  builder.Connect(offsets->get_output_port(), adder->get_input_port(1));

  // Connect the offset Sine reference to the IDC reference input.
  builder.Connect(adder->get_output_port(), smux->get_input_port(0));
  builder.Connect(sine_source->get_output_port(1), smux->get_input_port(1));
  builder.Connect(smux->get_output_port(0),
                  id_controller->get_input_port_desired_state());

  // Connect the ID controller directly (no translation)
  auto u2f = builder.AddSystem<MakePlantGeneralizedForceArray>(plant, plant_id);
  builder.Connect(id_controller->get_output_port_control(),
                  u2f->get_input_port(0));
  builder.Connect(u2f->get_output_port(0),
                  plant.get_applied_generalized_force_input_port());

  // Connect zero to actuation input port of MBP
  auto const_src = builder.AddSystem<systems::ConstantVectorSource>(
      VectorX<double>::Zero(2));
  builder.Connect(const_src->get_output_port(),
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
        plant.get_actuation_input_port().get_index(), tau_d);
  }

  // Set initial conditions.
  VectorX<double> gripper_ics = VectorX<double>::Zero(4);
  gripper_ics << -0.65, 1.21, 0, 0;

  // Finger 1
  const RevoluteJoint<double>& sh_pin1 =
      plant.GetJointByName<RevoluteJoint>("finger_ShoulderJoint");
  const RevoluteJoint<double>& el_pin1 =
      plant.GetJointByName<RevoluteJoint>("finger_ElbowJoint");
  sh_pin1.set_angle(&plant_context, gripper_ics(0));
  el_pin1.set_angle(&plant_context, gripper_ics(1));

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
