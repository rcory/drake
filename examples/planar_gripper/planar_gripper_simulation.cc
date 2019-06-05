#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/common/text_logging_gflags.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "Eigen/src/Core/Matrix.h"

namespace drake {
namespace examples {
namespace multibody {
namespace planar_gripper {
namespace {

using geometry::SceneGraph;
using lcm::DrakeLcm;

// "multibody" namespace is ambiguous here without "drake::".
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::multibody::RevoluteJoint;
using drake::math::RigidTransform;
using drake::math::RollPitchYaw;

DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");

DEFINE_double(simulation_time, 10.0,
              "Desired duration of the simulation in seconds.");

DEFINE_double(time_step, 1e-3,
            "If greater than zero, the plant is modeled as a system with "
            "discrete updates and period equal to this time_step. "
            "If 0, the plant is modeled as a continuous system.");

DEFINE_double(brick_z, 0, "Location of the brock on z-axis");

DEFINE_double(fix_input, false, "Fix the actuation inputs to zero?");

//MatrixX<double> GetKeyframes() {
//  MatrixX<double> keyframes()
//}

void WeldGripperFrames(MultibodyPlant<double>& plant) {
  const double kOuterRadius = 0.19;  // 19 - 22
  const double kF1Angle = 60 * (M_PI / 180.);
  const double kF23Angle = 120 * (M_PI / 180.);

  RigidTransform<double> XT(RollPitchYaw<double>(0, 0, 0),
                            Vector3<double>(0, 0, kOuterRadius));

  RigidTransform<double> X_PC1 =
      RigidTransform<double>(RollPitchYaw<double>(kF1Angle, 0, 0),
                             VectorX<double>::Zero(3)) * XT;
  plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("finger1_base"),
                   X_PC1);

  RigidTransform<double> X_PC2 =
      RigidTransform<double>(RollPitchYaw<double>(kF23Angle, 0, 0),
                             VectorX<double>::Zero(3)) * X_PC1;
  plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("finger2_base"),
                   X_PC2);

  RigidTransform<double> X_PC3 =
      RigidTransform<double>(RollPitchYaw<double>(kF23Angle, 0, 0),
                             VectorX<double>::Zero(3)) * X_PC2;
  plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("finger3_base"),
                   X_PC3);
}

class ActuatorTranslator : public systems::LeafSystem<double> {
 public:
  ActuatorTranslator(VectorX<double> ordering) : ordering_(ordering) {
    size_t size = ordering.size();

    this->DeclareVectorInputPort("input1", systems::BasicVector<double>(size));
    this->DeclareVectorOutputPort("output1", systems::BasicVector<double>(size),
                                  &ActuatorTranslator::reorder_output);
  }

  void reorder_output(const systems::Context<double>& context,
      systems::BasicVector<double>* output_vector) const {
    auto output_value = output_vector->get_mutable_value();
    DRAKE_DEMAND(output_value.size() == ordering_.size());

    auto input_value = this->EvalVectorInput(context, 0)->get_value();

    size_t size = ordering_.size();
    for (size_t i=0; i < size; i++) {
      output_value(i) = input_value(ordering_(i));
    }
  }

 private:
  VectorX<double> ordering_;
};

int do_main() {
  systems::DiagramBuilder<double> builder;

  SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
  scene_graph.set_name("scene_graph");

  // Make and add the planar_gripper model.
  const std::string full_name = FindResourceOrThrow(
      "drake/examples/planar_gripper/planar_gripper.sdf");
  MultibodyPlant<double>& plant =
      *builder.AddSystem<MultibodyPlant>(FLAGS_time_step);
  auto plant_id = Parser(&plant, &scene_graph).AddModelFromFile(full_name);
  WeldGripperFrames(plant);

  // Adds the object to be manipulated.
  auto object_file_name =
      FindResourceOrThrow("drake/examples/planar_gripper/brick.sdf");
  auto object_id = Parser(&plant).AddModelFromFile(object_file_name, "object");

  // Create the controlled plant. Contains only the fingers (no objects).
  MultibodyPlant<double> control_plant(FLAGS_time_step);
  Parser(&control_plant).AddModelFromFile(full_name);
  WeldGripperFrames(control_plant);

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
  VectorX<double> Kp(6), Kd(6), Ki(6);
  Kp = VectorX<double>::Ones(6) * 1500;
  Ki = VectorX<double>::Ones(6) * 500;
  Kd = VectorX<double>::Ones(6) * 500;

  auto id_controller =
      builder.AddSystem<systems::controllers::InverseDynamicsController>(
          control_plant, Kp, Ki, Kd, false);

  // Connect the ID controller
  builder.Connect(plant.get_state_output_port(plant_id),
                  id_controller->get_input_port_estimated_state());

  // Constant reference.
  VectorX<double> x_ref = VectorX<double>::Zero(12);

  // Connect the desired state
  auto const_src = builder.AddSystem<systems::ConstantVectorSource>(x_ref);
  builder.Connect(const_src->get_output_port(),
                  id_controller->get_input_port_desired_state());

  // TODO(rcory) This connect code doesn't work...seems indices don't match.
  // builder.Connect(id_controller.get_output_port_control(),
  //                 plant.get_actuation_input_port())
  //
  // Hack needed to map the ID controller outputs to MBP inputs.
  // TODO(rcory) Update ID controller to be "smarter" and know about the
  //  required index actuator ordering going into MBP.
  VectorX<double> ordering(6);
  ordering << 0, 3, 1, 4, 2, 5;
  auto translator = builder.AddSystem<ActuatorTranslator>(ordering);
  builder.Connect(id_controller->get_output_port_control(),
                  translator->get_input_port(0));
  builder.Connect(translator->get_output_port(0),
                  plant.get_actuation_input_port(plant_id));

//  // Connect the ID controller directly (no translation)
//  builder.Connect(id_controller->get_output_port_control(),
//                  plant.get_actuation_input_port(plant_id));

  // Connect MBP snd SG.
  builder.Connect(
      plant.get_geometry_poses_output_port(),
      scene_graph.get_source_pose_port(plant.get_source_id().value()));
  builder.Connect(scene_graph.get_query_output_port(),
      plant.get_geometry_query_input_port());

  geometry::ConnectDrakeVisualizer(&builder, scene_graph);
  auto diagram = builder.Build();

  // Create a context for this system:
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  // There is no input actuation in this example for the passive dynamics.
  if (FLAGS_fix_input) {
    VectorX<double> tau_d(6);
    tau_d << 0, 0, 0, 0, 0, 0;
    plant_context.FixInputPort(
        plant.get_actuation_input_port().get_index(), tau_d);
  }

  // Set initial conditions.
  // Finger 1
  const RevoluteJoint<double>& sh_pin1 =
      plant.GetJointByName<RevoluteJoint>("finger1_ShoulderJoint");
  const RevoluteJoint<double>& el_pin1 =
      plant.GetJointByName<RevoluteJoint>("finger1_ElbowJoint");
  sh_pin1.set_angle(&plant_context, -0.65);
  el_pin1.set_angle(&plant_context, 1.0);

  // Finger 2
  const RevoluteJoint<double>& sh_pin2 =
      plant.GetJointByName<RevoluteJoint>("finger2_ShoulderJoint");
  const RevoluteJoint<double>& el_pin2 =
      plant.GetJointByName<RevoluteJoint>("finger2_ElbowJoint");
  sh_pin2.set_angle(&plant_context, -0.5);
  el_pin2.set_angle(&plant_context, 0.95);

  // Finger 3
  const RevoluteJoint<double>& sh_pin3 =
      plant.GetJointByName<RevoluteJoint>("finger3_ShoulderJoint");
  const RevoluteJoint<double>& el_pin3 =
      plant.GetJointByName<RevoluteJoint>("finger3_ElbowJoint");
  sh_pin3.set_angle(&plant_context, 0.65);
  el_pin3.set_angle(&plant_context, -1.0);

  // Set the box initial conditions.
  RigidTransform<double> X_WObj(
      RollPitchYaw<double>(0, 0, 0), Vector3<double>(0, 0, FLAGS_brick_z));
  auto body_index_vec = plant.GetBodyIndices(object_id);
  auto& box_body = plant.get_body(body_index_vec[0]);
  plant.SetFreeBodyPose(&plant_context, box_body, X_WObj);

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

  simulator.set_publish_every_time_step(false);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);

  return 0;
}

}  // namespace
}  // namespace planar_gripper
}  // namespace multibody
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "A simple planar gripper example.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::logging::HandleSpdlogGflags();
  return drake::examples::multibody::planar_gripper::do_main();
}
