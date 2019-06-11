#include <memory>

#include <gflags/gflags.h>

#include "robotlocomotion/robot_plan_t.hpp"
#include "drake/examples/planar_gripper/planar_gripper_common.h"

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/common/text_logging_gflags.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/manipulation/planner/robot_plan_interpolator.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/constant_value_source.h"
#include "Eigen/src/Core/Matrix.h"

namespace drake {
namespace examples {
namespace planar_gripper {
namespace {

using geometry::SceneGraph;

using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::multibody::RevoluteJoint;
using drake::math::RigidTransform;
using drake::math::RollPitchYaw;
using drake::manipulation::planner::RobotPlanInterpolator;
using drake::multibody::JointActuatorIndex;

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
DEFINE_double(keyframe_dt, 0.1, "keyframe dt");


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
  examples::planar_gripper::WeldGripperFrames(plant);

  // Adds the object to be manipulated.
  auto object_file_name =
      FindResourceOrThrow("drake/examples/planar_gripper/brick.sdf");
  auto object_id = Parser(&plant).AddModelFromFile(object_file_name, "object");

  // Create the controlled plant. Contains only the fingers (no objects).
  MultibodyPlant<double> control_plant(FLAGS_time_step);
  Parser(&control_plant).AddModelFromFile(full_name);
  drake::examples::planar_gripper::WeldGripperFrames(control_plant);

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

  // == Keyframe Interpolator ================
  // Read the keyframes from a file.
  const std::string keyframe_path =
      "drake/examples/planar_gripper/postures.txt";

  Vector3<double> brick_ics;
  auto keyframes = examples::planar_gripper::ParseKeyframes(keyframe_path, &brick_ics);

  keyframes.transposeInPlace();

  // Creates the time vector.
  std::vector<double> times(keyframes.cols());
  const double kDt = FLAGS_keyframe_dt;
  times[0] = 0.0;
  for (int i = 1; i < keyframes.cols(); ++i) {
    times[i] = times[i-1] + kDt;
  }
  std::vector<int> info(times.size(), 1);
  robotlocomotion::robot_plan_t plan{};
  plan = manipulation::planner::EncodeKeyFrames(control_plant, times, info,
                                                keyframes);

  // Publish the plan for inspection
  examples::planar_gripper::PublishRobotPlan(plan);

  auto interp = builder.AddSystem<RobotPlanInterpolator>(
      control_plant, manipulation::planner::InterpolatorType::Pchip,
      kDt);
  // Constant reference **AbstractValue** source
  std::unique_ptr<AbstractValue> avalue =
      AbstractValue::Make(plan);
  auto const_value_src =
      builder.AddSystem<systems::ConstantValueSource<double>>(*avalue);
  builder.Connect(const_value_src->get_output_port(0),
                  interp->get_plan_input_port());
  builder.Connect(interp->get_state_output_port(),
                  id_controller->get_input_port_desired_state());
  // =============================================

//    // Constant reference **vector** source.
//    VectorX<double> x_ref = VectorX<double>::Zero(12);
//    x_ref << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
//
//    // Connect the reference
//    auto const_src = builder.AddSystem<systems::ConstantVectorSource>(x_ref);
//    builder.Connect(const_src->get_output_port(),
//                    id_controller->get_input_port_desired_state());

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

  if (FLAGS_fix_input) {
    VectorX<double> tau_d(6);
    tau_d << 0, 0, 0, 0, 0, 0;
    plant_context.FixInputPort(
        plant.get_actuation_input_port().get_index(), tau_d);
  }

  // Set initial conditions.
  VectorX<double> gripper_ics = VectorX<double>::Zero(12);
  gripper_ics.head(6) = keyframes.block(0, 0, 6, 1);

  // Finger 1
  const RevoluteJoint<double>& sh_pin1 =
      plant.GetJointByName<RevoluteJoint>("finger1_ShoulderJoint");
  const RevoluteJoint<double>& el_pin1 =
      plant.GetJointByName<RevoluteJoint>("finger1_ElbowJoint");
  sh_pin1.set_angle(&plant_context, gripper_ics(0));
  el_pin1.set_angle(&plant_context, gripper_ics(3));

  // Finger 2
  const RevoluteJoint<double>& sh_pin2 =
      plant.GetJointByName<RevoluteJoint>("finger2_ShoulderJoint");
  const RevoluteJoint<double>& el_pin2 =
      plant.GetJointByName<RevoluteJoint>("finger2_ElbowJoint");
  sh_pin2.set_angle(&plant_context, gripper_ics(1));
  el_pin2.set_angle(&plant_context, gripper_ics(4));

  // Finger 3
  const RevoluteJoint<double>& sh_pin3 =
      plant.GetJointByName<RevoluteJoint>("finger3_ShoulderJoint");
  const RevoluteJoint<double>& el_pin3 =
      plant.GetJointByName<RevoluteJoint>("finger3_ElbowJoint");
  sh_pin3.set_angle(&plant_context, gripper_ics(2));
  el_pin3.set_angle(&plant_context, gripper_ics(5));

  // Set the box initial conditions.
  Vector3<double> brick_pos(0, brick_ics(0), brick_ics(1) + FLAGS_brick_z);
  RigidTransform<double> X_WObj(
      RollPitchYaw<double>(brick_ics(2), 0, 0), brick_pos);
  auto body_index_vec = plant.GetBodyIndices(object_id);
  auto& box_body = plant.get_body(body_index_vec[0]);
  plant.SetFreeBodyPose(&plant_context, box_body, X_WObj);

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

  // Initialize the interpolator.
  auto& dcontext = simulator.get_mutable_context();
  auto& interpolator_context =
      diagram->GetMutableSubsystemContext(*interp, &dcontext);
  interp->Initialize(0 /* plan start time */, gripper_ics.head(6),
                     &interpolator_context.get_mutable_state());

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
