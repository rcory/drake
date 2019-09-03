/// @file
///
/// This demo sets up a simple dynamic simulation for the Allegro hand using
/// the multi-body library with inverse dynamics control.

#include <gflags/gflags.h>
#include <fstream>

#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/common/text_logging_gflags.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/math/rotation_matrix.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/multibody/tree/uniform_gravity_field_element.h"
#include "drake/multibody/tree/weld_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/signal_logger.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/examples/allegro_hand/allegro_common.h"
#include "drake/examples/allegro_hand/allegro_lcm.h"
#include "drake/lcmt_allegro_command.hpp"
#include "drake/lcmt_allegro_status.hpp"
#include "drake/systems/primitives/first_order_low_pass_filter.h"
#include "drake/systems/primitives/multiplexer.h"
#include "drake/systems/primitives/demultiplexer.h"

namespace drake {
namespace examples {
namespace allegro_hand {

using drake::multibody::MultibodyPlant;
using drake::multibody::ModelInstanceIndex;
using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;

DEFINE_double(simulation_time, std::numeric_limits<double>::infinity(),
"Desired duration of the simulation in seconds");

DEFINE_double(time_constant, 0.085,
"Time constant for actuator delay");

DEFINE_bool(use_right_hand, true,
"Which hand to model: true for right hand or false for left hand");

DEFINE_double(kp, 1000000,
"Proportional control gain for all joints, 1000000 seems good");

DEFINE_double(ki, 0,
"Integral control gain for all joints, 0 seems good");

DEFINE_double(kd, 1500,
"Derivative control gain for all joints, 1500 seems good");

DEFINE_double(max_time_step, 5.0e-4,
"Simulation time step used for integrator.");

DEFINE_bool(add_gravity, true, "Indicator for whether terrestrial gravity"
" (9.81 m/s²) is included or not.");

DEFINE_double(target_realtime_rate, 1,
"Desired rate relative to real time.  See documentation for "
"Simulator::set_target_realtime_rate() for details.");

DEFINE_double(floor_coef_static_friction, 0,
        "Coefficient of static friction for the floor.");

DEFINE_double(floor_coef_kinetic_friction, 0,
        "Coefficient of kinetic friction for the floor. "
        "When time_step > 0, this value is ignored. Only the "
        "coefficient of static friction is used in fixed-time step.");

/// Maps a user state xₛ to the MPB state x, based on the preferred ordering
/// defined in allegro_common.cc
class DesiredStateToIDCRemap : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DesiredStateToIDCRemap);
  DesiredStateToIDCRemap(const MultibodyPlant<double>& control_plant) {
    this->DeclareVectorInputPort(
        "input",
        systems::BasicVector<double>(control_plant.num_velocities() * 2));
    this->DeclareVectorOutputPort(
        "output",
        systems::BasicVector<double>(control_plant.num_velocities() * 2),
        &DesiredStateToIDCRemap::remap_output);

    // Get the state/actuation mapping for the control plant.
    MatrixX<double> Sx, Sy;
    GetControlPortMapping(control_plant, &Sx, &Sy);
    Sx_inverse_ = Sx.inverse();
  }

  void remap_output(const systems::Context<double>& context,
                    systems::BasicVector<double>* output_vector) const {
    auto output_value = output_vector->get_mutable_value();
    auto input_value = this->EvalVectorInput(context, 0)->get_value();
    output_value = Sx_inverse_ * input_value;
  }

 private:
  MatrixX<double> Sx_inverse_;
};

/// Remaps the input vector (which is mapped using the code's torque vector
/// mapping and corresponds to the generalized force output of the ID controller
/// using a control plant with only the hand) into the output vector (which is
/// mapped using the code's position vector mapping and corresponds to the
/// generalized force input for the full simulation plant containing hand and
/// object).
class MapTorqueToPositionVector : public systems::LeafSystem<double> {
public:
    DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MapTorqueToPositionVector);
    MapTorqueToPositionVector(const MultibodyPlant<double>* plant,
                              ModelInstanceIndex gripper_instance)
            : plant_(plant), gripper_instance_(gripper_instance) {
        DRAKE_DEMAND(plant != nullptr);
        this->DeclareVectorInputPort(
                "input", systems::BasicVector<double>(plant->num_actuators()));
        this->DeclareVectorOutputPort(
                "output", systems::BasicVector<double>(plant->num_velocities()),
                &MapTorqueToPositionVector::remap_output);
    }

    void remap_output(const systems::Context<double>& context,
                      systems::BasicVector<double>* output_vector) const {
      auto output_value = output_vector->get_mutable_value();
      output_value.setZero();
      auto input_value = this->EvalVectorInput(context, 0)->get_value();
      VectorX<double> hand_actuator_values(kAllegroNumJoints);
      plant_->SetVelocitiesInArray(gripper_instance_, input_value,
                                   &output_value);
    }

   private:
    const MultibodyPlant<double>* plant_;
    const ModelInstanceIndex gripper_instance_;
};


void DoMain() {
    DRAKE_DEMAND(FLAGS_simulation_time > 0);

    systems::DiagramBuilder<double> builder;
    auto lcm = builder.AddSystem<systems::lcm::LcmInterfaceSystem>();

    geometry::SceneGraph<double>& scene_graph =
            *builder.AddSystem<geometry::SceneGraph>();
    scene_graph.set_name("scene_graph");

    MultibodyPlant<double>& plant =
        *builder.AddSystem<MultibodyPlant>(FLAGS_max_time_step);
    plant.RegisterAsSourceForSceneGraph(&scene_graph);
    std::string hand_model_path;
    if (FLAGS_use_right_hand)
      hand_model_path = FindResourceOrThrow(
          "drake/manipulation/models/"
          "allegro_hand_description/sdf/allegro_hand_description_right.sdf");
    else
      hand_model_path = FindResourceOrThrow(
          "drake/manipulation/models/"
          "allegro_hand_description/sdf/allegro_hand_description_left.sdf");

    const std::string object_model_path = FindResourceOrThrow(
        "drake/examples/allegro_hand/teleop_manus_attempt1/block.sdf");

    multibody::Parser parser(&plant);
    const ModelInstanceIndex hand_index =
        parser.AddModelFromFile(hand_model_path);
    parser.AddModelFromFile(object_model_path);

    // Weld the hand to the world frame
    const auto& joint_hand_root = plant.GetBodyByName("hand_root");
    plant.AddJoint<multibody::WeldJoint>("weld_hand", plant.world_body(),
                                         nullopt, joint_hand_root, nullopt,
                                         math::RigidTransformd::Identity());

    if (!FLAGS_add_gravity) {
      plant.mutable_gravity_field().set_gravity_vector(Eigen::Vector3d::Zero());
    }

    // Create the controlled plant. Contains only the hand (no objects).
    MultibodyPlant<double> control_plant(FLAGS_max_time_step);
    multibody::Parser(&control_plant).AddModelFromFile(hand_model_path);
    control_plant.AddJoint<multibody::WeldJoint>(
        "weld_hand", control_plant.world_body(), nullopt,
        control_plant.GetBodyByName("hand_root"), nullopt,
        math::RigidTransformd::Identity());

    // Add a floor (an infinite halfspace) to the plant
    const Vector4<double> color(1.0, 1.0, 1.0, 1.0);
    const drake::multibody::CoulombFriction<double> coef_friction_floor(
            FLAGS_floor_coef_static_friction,
            FLAGS_floor_coef_kinetic_friction);
    plant.RegisterVisualGeometry(plant.world_body(),
            math::RigidTransformd::Identity(),
            geometry::HalfSpace(), "FloorVisualGeometry", color);
    plant.RegisterCollisionGeometry(plant.world_body(),
            math::RigidTransformd::Identity(),
            geometry::HalfSpace(), "InclinedPlaneCollisionGeometry",
            coef_friction_floor);

    // Now the plant is complete.
    plant.Finalize();
    control_plant.Finalize();

    DRAKE_DEMAND(plant.num_actuators() == 16);
    DRAKE_DEMAND(plant.num_actuated_dofs() == 16);

    // Add inverse dynamics controller
    auto IDC =
        builder
            .AddSystem<systems::controllers::InverseDynamicsController<double>>(
                control_plant,
                Eigen::VectorXd::Ones(kAllegroNumJoints) * FLAGS_kp,
                Eigen::VectorXd::Ones(kAllegroNumJoints) * FLAGS_ki,
                Eigen::VectorXd::Ones(kAllegroNumJoints) * FLAGS_kd, false);

    // Add low pass filter block
    auto filter = builder.AddSystem<systems::FirstOrderLowPassFilter<double>>(
            FLAGS_time_constant, 2*kAllegroNumJoints);

    // Add demultiplexer to pass only first elements of remap system output to
    // status sender
    std::vector<int> output_sizes = {kAllegroNumJoints, 6};
    auto demultiplexer =
        builder.AddSystem<systems::Demultiplexer<double>>(output_sizes);

    // Create the status publisher and sender to log hand info so it is visible
    // on LCM Spy
    auto& hand_status_pub = *builder.AddSystem(
        systems::lcm::LcmPublisherSystem::Make<lcmt_allegro_status>(
            "ALLEGRO_STATUS", lcm, kLcmStatusPeriod /* publish period */));
    hand_status_pub.set_name("hand_status_publisher");
    auto& status_sender =
        *builder.AddSystem<AllegroStatusSender>(kAllegroNumJoints);
    status_sender.set_name("status_sender");

    // Add system to remap control ports to match indexing of state ports
    auto remap_sys =
        builder.AddSystem<MapTorqueToPositionVector>(&plant, hand_index);

    // Create a constant zero vector to connect to the actuation input port of
    // MBP since we don't use it (we use the generalized forces input).
    auto const_src = builder.AddSystem<systems::ConstantVectorSource>(
        VectorX<double>::Zero(kAllegroNumJoints));

    // Create the command subscriber for the hand.
    auto& hand_command_sub = *builder.AddSystem(
        systems::lcm::LcmSubscriberSystem::Make<lcmt_allegro_command>(
            "ALLEGRO_COMMAND", lcm));
    hand_command_sub.set_name("hand_command_subscriber");
    auto& hand_command_receiver =
            *builder.AddSystem<AllegroCommandReceiver>(kAllegroNumJoints);
    hand_command_receiver.set_name("hand_command_receiver");

    // Add signal loggers to log system desired and actual state to a file
    auto desired_state_logger = LogOutput(
        hand_command_receiver.get_commanded_state_output_port(), &builder);
    auto actual_state_logger =
        LogOutput(plant.get_state_output_port(hand_index), &builder);

    // A system to remap the incoming state input to the IDC.
    auto desired_state_remap =
        builder.AddSystem<DesiredStateToIDCRemap>(control_plant);

    // Connect ports
    builder.Connect(*IDC, *remap_sys);
    builder.Connect(remap_sys->get_output_port(0),
                    plant.get_applied_generalized_force_input_port());
    builder.Connect(const_src->get_output_port(),
                    plant.get_actuation_input_port());
    builder.Connect(plant.get_state_output_port(hand_index),
                    IDC->get_input_port_estimated_state());
    builder.Connect(filter->get_output_port(),
                    desired_state_remap->get_input_port(0));
    builder.Connect(desired_state_remap->get_output_port(0),
                    IDC->get_input_port_desired_state());
    builder.Connect(remap_sys->get_output_port(0),
                    demultiplexer->get_input_port(0));
    builder.Connect(demultiplexer->get_output_port(0),
                    status_sender.get_commanded_torque_input_port());
    builder.Connect(plant.get_state_output_port(hand_index),
                    status_sender.get_state_input_port());
    builder.Connect(status_sender.get_output_port(0),
                    hand_status_pub.get_input_port());
    builder.Connect(hand_command_sub.get_output_port(),
                    hand_command_receiver.get_input_port(0));
    builder.Connect(hand_command_receiver.get_commanded_state_output_port(),
                    filter->get_input_port());
    builder.Connect(hand_command_receiver.get_commanded_state_output_port(),
                    status_sender.get_command_input_port());

    // Connect scenegraph and drake visualizer
    geometry::ConnectDrakeVisualizer(&builder, scene_graph);
    DRAKE_DEMAND(!!plant.get_source_id());
    builder.Connect(
        plant.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id().value()));
    builder.Connect(scene_graph.get_query_output_port(),
                    plant.get_geometry_query_input_port());

    // Build diagram
    std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();

    geometry::DispatchLoadMessage(scene_graph, lcm);

    // Create a context for this system
    std::unique_ptr<systems::Context<double>> diagram_context =
            diagram->CreateDefaultContext();
    diagram->SetDefaultContext(diagram_context.get());
    systems::Context<double>& plant_context =
            diagram->GetMutableSubsystemContext(plant, diagram_context.get());

    // Set initial conditions for block
    const multibody::Body<double>& block = plant.GetBodyByName("main_body");
    const multibody::Body<double>& hand = plant.GetBodyByName("hand_root");
    const Eigen::Vector3d& p_WHand =
            plant.EvalBodyPoseInWorld(plant_context, hand).translation();
    RigidTransformd X_WM(
            RollPitchYawd(M_PI / 2, 0, 0),
            p_WHand + Eigen::Vector3d(0.095, 0.062, 0.095));
    plant.SetFreeBodyPose(&plant_context, block, X_WM);

    // Set up simulator.
    systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
    simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
    simulator.Initialize();
    simulator.AdvanceTo(FLAGS_simulation_time);

    // Print logged data to file
    // Gets the time stamps when each data point is saved.
    const auto& desired_times = desired_state_logger->sample_times();
    // Gets the logged data.
    const auto& desired_data = desired_state_logger->data();
    // Gets the time stamps when each data point is saved.
    const auto& actual_times = actual_state_logger->sample_times();
    // Gets the logged data.
    const auto& actual_data = actual_state_logger->data();

    std::fstream outfile;
    outfile.open("test.txt", std::fstream::out);
    outfile << desired_times.transpose() << std::endl;
    outfile << desired_data << std::endl;
    outfile << actual_times.transpose() << std::endl;
    outfile << actual_data << std::endl;
    outfile.close();
}

}  // namespace allegro_hand
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
    gflags::SetUsageMessage(
            "A dynamic simulation for the Allegro hand.");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    drake::examples::allegro_hand::DoMain();
    return 0;
}