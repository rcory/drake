#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/examples/planar_gripper/finger_brick.h"
#include "drake/examples/planar_gripper/finger_brick_control.h"
#include "drake/examples/planar_gripper/planar_gripper_utils.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/plant/spatial_forces_to_lcm.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_value_source.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/zero_order_hold.h"

namespace drake {
namespace examples {
namespace planar_gripper {

using Eigen::VectorXd;
using Eigen::Vector3d;
using geometry::SceneGraph;
using multibody::AddMultibodyPlantSceneGraph;
using multibody::ContactResults;
using multibody::MultibodyPlant;
using multibody::SpatialForce;

class SpatialForceVizSrc final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SpatialForceVizSrc);
  SpatialForceVizSrc(const MultibodyPlant<double>& plant, BrickType brick_type)
      : plant_(plant), brick_type_(brick_type) {
    this->DeclareVectorInputPort("F_Bq_W", systems::BasicVector<double>(3));
    this->DeclareAbstractInputPort("contact_results",
                                   Value<multibody::ContactResults<double>>{});

    // This output port produces a SpatialForceOutput, which feeds the spatial
    // forces visualization plugin of DrakeVisualizer.
    this->DeclareAbstractOutputPort(&SpatialForceVizSrc::SetOutput);
  }

 private:
  void SetOutput(const systems::Context<double>& context,
                 std::vector<multibody::SpatialForceOutput<double>>*
                     spatial_force_viz_output) const {
    spatial_force_viz_output->clear();
    Vector3d F_Bq_W = this->EvalVectorInput(context, 0)->get_value();
    const auto& contact_results =
        this->GetInputPort("contact_results")
            .Eval<multibody::ContactResults<double>>(context);
    if (brick_type_ == BrickType::FixedBrick) {
      // Should have no more than 1 contact pair in the sim.
      DRAKE_DEMAND(contact_results.num_point_pair_contacts() <= 1);
    } else if (brick_type_ == BrickType::PlanarBrick) {
      // Should have no more than 6 contact pairs in the sim (five brick/floor
      // corner contacts, and one fingertip/brick contact).
      DRAKE_DEMAND(contact_results.num_point_pair_contacts() <= 6);
    }
    // extract finger/brick contact pair index.
    std::optional<int> pair_index =
        GetContactPairIndex(plant_, contact_results, Finger::kFinger1);
    if (pair_index.has_value()) {
      Vector3d p_BoBq_W =
          contact_results.point_pair_contact_info(pair_index.value())
              .contact_point();
      SpatialForce<double> SF_Bq_W(Vector3d::Zero(), F_Bq_W);
      spatial_force_viz_output->emplace_back(
          multibody::SpatialForceOutput(p_BoBq_W, SF_Bq_W));
    }
  }

  const MultibodyPlant<double>& plant_;
  const BrickType brick_type_;
};

/// This is a ForceController integration test. The plant consists of a single
/// planar finger and a fixed brick, and the test checks the ForceController's
/// ability to regulate a fixed force against the surface of the brick.
class ForceControlTest : public testing::Test {
 protected:
  void BuildDiagram(BrickType brick_type,
                    Vector3d brick_pose /* (y, z, theta) */,
                    Vector2d desired_force /* (fy, fz) */,
                    Vector2d finger_q0 /* (base, mid) */,
                    double finger_weld_angle) {
    systems::DiagramBuilder<double> builder;
    MultibodyPlant<double>* plant;
    SceneGraph<double>* scene_graph;
    std::tie(plant, scene_graph) = AddMultibodyPlantSceneGraph(
        &builder, std::make_unique<MultibodyPlant<double>>(1e-3),
        std::make_unique<SceneGraph<double>>());

    // Add the planar_finger model.
    const std::string finger_filename = FindResourceOrThrow(
        "drake/examples/planar_gripper/models/planar_finger.sdf");
    auto finger_index = multibody::Parser(plant, scene_graph)
                             .AddModelFromFile(finger_filename);
    WeldFingerFrame<double>(plant, finger_weld_angle);

    // Adds a single fixed brick (specifically for force control testing).
    std::string brick_filename;
    switch (brick_type) {
      case BrickType::FixedBrick:
        brick_filename = "drake/examples/planar_gripper/models/fixed_brick.sdf";
        break;
      case BrickType::PlanarBrick:
        brick_filename =
            "drake/examples/planar_gripper/models/planar_brick.sdf";
        break;
      default:
        throw std::logic_error("Unsupported brick type.");
    }
    auto object_file_name = FindResourceOrThrow(brick_filename);
    auto brick_index = multibody::Parser(plant, scene_graph)
                           .AddModelFromFile(object_file_name, "brick");

    const multibody::Frame<double>& brick_base_frame =
        plant->GetFrameByName("brick_base_link", brick_index);
    math::RigidTransformd weld_xform = math::RigidTransformd();
    if (brick_type == BrickType::FixedBrick) {
      weld_xform.set_translation(
          Eigen::Vector3d(0, brick_pose(0), brick_pose(1)));
      weld_xform.set_rotation(math::RollPitchYaw<double>(brick_pose(2), 0, 0));
    } else {  // PlanarBrick, so we add a floor.
      double kFloorHeight = 0.025;
      double kBrickWidth = 0.07;
      double kPenetration = 1e-4;
      const math::RigidTransformd X_WF(
          Eigen::AngleAxisd(M_PI_2, Vector3d::UnitY()),
          Vector3d((-kFloorHeight - kBrickWidth) / 2.0 + kPenetration,
                   brick_pose(0), brick_pose(1)));
      const Vector4<double> white(0.8, 0.8, 0.8, 0.6);
      plant->RegisterVisualGeometry(plant->world_body(), X_WF,
                                    geometry::Cylinder(.125, kFloorHeight),
                                    "FloorVisualGeometry", white);
      const drake::multibody::CoulombFriction<double> coef_friction_floor(
          0.5 /* static fric coef. */, 0.5 /* kinetic friction coef */);
      plant->RegisterCollisionGeometry(
          plant->world_body(), X_WF, geometry::Cylinder(.125, kFloorHeight),
          "FloorCollisionGeometry", coef_friction_floor);
    }
    plant->WeldFrames(plant->world_frame(), brick_base_frame, weld_xform);

    plant->Finalize();

    plant->set_penetration_allowance(0.2);
    plant->set_stiction_tolerance(1e-3);

    geometry::ConnectDrakeVisualizer(&builder, *scene_graph, &drake_lcm_);
    ConnectContactResultsToDrakeVisualizer(
        &builder, *plant, plant->get_contact_results_output_port());

    // Setup the force control parameters. These values are experimentally known
    // to work well in simulation.
    ForceControlOptions foptions;
    foptions.kpf_t_ = 850;
    foptions.kpf_n_ = 5e3;
    foptions.kif_t_ = 5e3;
    foptions.kif_n_ = 1e2;
    foptions.kp_t_ = 0;
    foptions.kd_t_ = 2e3;
    foptions.kp_n_ = 0;
    foptions.kd_n_ = 15e3;
    foptions.Kd_joint_ << 1.0, 0, 0, 1.0;
    foptions.K_compliance_ = 2e3;
    foptions.D_damping_ = 1e3;
    foptions.always_direct_force_control_ = false;
    foptions.finger_to_control_ = Finger::kFinger1;

    // Connect finger/plant states to force controller.
    force_controller_ =
        builder.AddSystem<ForceController>(*plant, *scene_graph, foptions);
    builder.Connect(plant->get_state_output_port(finger_index),
                    force_controller_->get_finger_state_actual_input_port());
    builder.Connect(plant->get_state_output_port(),
                    force_controller_->get_plant_state_actual_input_port());

    // Connect the "virtual" force sensor to the force controller.
    auto zoh_contact_results =
        builder.AddSystem<systems::ZeroOrderHold<double>>(
            1e-3, Value<ContactResults<double>>());
    builder.Connect(plant->get_contact_results_output_port(),
                    zoh_contact_results->get_input_port());
    std::vector<SpatialForce<double>> init_spatial_vec{
        SpatialForce<double>(Vector3<double>::Zero(), Vector3<double>::Zero())};
    auto zoh_reaction_forces =
        builder.AddSystem<systems::ZeroOrderHold<double>>(
            1e-3, Value<std::vector<SpatialForce<double>>>(init_spatial_vec));
    builder.Connect(plant->get_reaction_forces_output_port(),
                    zoh_reaction_forces->get_input_port());
    auto force_demux_sys =
        builder.AddSystem<ForceDemuxer>(*plant, foptions.finger_to_control_);
    builder.Connect(zoh_contact_results->get_output_port(),
                    force_demux_sys->get_contact_results_input_port());
    builder.Connect(zoh_reaction_forces->get_output_port(),
                    force_demux_sys->get_reaction_forces_input_port());
    builder.Connect(plant->get_state_output_port(),
                    force_demux_sys->get_state_input_port());
    builder.Connect(force_demux_sys->get_reaction_vec_output_port(),
                    force_controller_->get_force_sensor_input_port());

    // Connects the desired fingertip state to the force controller. Currently
    // desired position is unused by the force controller (for now).
    // Additionally, the desired velocity of the contact point is always zero
    // (strict damping), so we set the entire state vector to zero. The
    // 6-element vector represents positions and velocities for the fingertip
    // contact point x-y-z. The controller internally ignores the x-components.
    auto tip_state_desired_src =
        builder.AddSystem<systems::ConstantVectorSource>(
            Vector6<double>::Zero());
    builder.Connect(tip_state_desired_src->get_output_port(),
                    force_controller_->get_contact_state_desired_input_port());

    // Connect the force controller to the plant.
    builder.Connect(force_controller_->get_torque_output_port(),
                    plant->get_actuation_input_port());

    // Create a source for the desired force.  The actual force chosen here for
    // this test is arbitrary, with magnitudes on the same scale as what is seen
    // for the brick rotation simulation.
    multibody::ExternallyAppliedSpatialForce<double> desired_spatial_force;
    desired_spatial_force.F_Bq_W = multibody::SpatialForce<double>(
        Vector3d::Zero() /* torque */,
        Vector3d(0, desired_force(0), desired_force(1)) /* force */);

    auto desired_force_src = builder.AddSystem<
        systems::ConstantValueSource<double>>(
        Value<std::vector<multibody::ExternallyAppliedSpatialForce<double>>>(
            {desired_spatial_force}));
    builder.Connect(desired_force_src->get_output_port(0),
                    force_controller_->get_force_desired_input_port());

    // Draw the desired force vector in drake visualizer.
    auto force_viz_src =
        builder.AddSystem<SpatialForceVizSrc>(*plant, brick_type);
    auto const_force_src = builder.AddSystem<systems::ConstantVectorSource>(
        Vector3d(0, desired_force(0), desired_force(1)));
    builder.Connect(const_force_src->get_output_port(),
                    force_viz_src->GetInputPort("F_Bq_W"));
    builder.Connect(plant->get_contact_results_output_port(),
                    force_viz_src->GetInputPort("contact_results"));
    multibody::ConnectSpatialForcesToDrakeVisualizer(
        &builder, *plant, force_viz_src->get_output_port(0), &drake_lcm_);

    // Note, body index and contact point are not (currently) used in the force
    // controller. We set them here anyway.
    desired_spatial_force.body_index =
        plant->GetBodyByName("brick_link").index();
    desired_spatial_force.p_BoBq_B = Vector3d::Zero();

    // Add the system which calculates the brick's contact face &
    // witness/contact point.
    const std::vector<Finger> fingers = {foptions.finger_to_control_};
    auto finger_face_assigner =
        builder.AddSystem<FingerFaceAssigner>(*plant, *scene_graph, fingers);
    builder.Connect(zoh_contact_results->get_output_port(),
                    finger_face_assigner->GetInputPort("contact_results"));
    builder.Connect(scene_graph->get_query_output_port(),
                    finger_face_assigner->GetInputPort("geometry_query"));
    builder.Connect(plant->get_state_output_port(),
                    finger_face_assigner->GetInputPort("plant_state"));
    builder.Connect(
        finger_face_assigner->GetOutputPort("finger_face_assignments"),
        force_controller_->GetInputPort("finger_face_assignments"));

    diagram_ = builder.Build();

    diagram_context_ = diagram_->CreateDefaultContext();
    systems::Context<double>* plant_context =
        &diagram_->GetMutableSubsystemContext(*plant, diagram_context_.get());

    // Set arbitrary initial joint positions.
    plant->SetPositions(plant_context, finger_index, finger_q0);

    // If the brick isn't fixed, set the initial pose.
    if (brick_type != BrickType::FixedBrick) {
      plant->SetPositions(plant_context, brick_index, brick_pose);
      plant->SetVelocities(plant_context, brick_index, Eigen::Vector3d::Zero());
    }

    // Add standard gravity in the -z axis.
    plant->mutable_gravity_field().set_gravity_vector(Vector3d(
        0, 0,
        -multibody::UniformGravityFieldElement<double>::kDefaultStrength*0));
  }

  Eigen::Vector2d Simulate() {
    // Simulate the system.
    systems::Simulator<double> simulator(*diagram_,
                                         std::move(diagram_context_));
    simulator.set_target_realtime_rate(1);
    simulator.Initialize();
    simulator.AdvanceTo(2);  // Run to approximate convergence.

    // Check the steady state actual force against the desired force.
    const auto& post_sim_context = simulator.get_context();
    const auto& post_force_controller_context =
        diagram_->GetSubsystemContext(*force_controller_, post_sim_context);
    const Eigen::Vector3d F_Bq_W_actual =
        force_controller_
            ->EvalVectorInput(
                post_force_controller_context,
                force_controller_->get_force_sensor_input_port().get_index())
            ->get_value();
    return F_Bq_W_actual.tail<2>();
  }

  std::unique_ptr<systems::Diagram<double>> diagram_;
  std::unique_ptr<systems::Context<double>> diagram_context_;
  ForceController* force_controller_{nullptr};
  lcm::DrakeLcm drake_lcm_;
};

TEST_F(ForceControlTest, PlanarFingerStaticForceControlPosZ) {
  // Build the diagram and simulate the system.
  Vector2d desired_force(-0.032, -0.065);
  BuildDiagram(BrickType::FixedBrick, Vector3d::Zero() /* brick pose */,
               desired_force, Vector2d(0.5, -1.4) /* finger q0 */,
               0 /* finger weld angle */);
  const Eigen::Vector2d final_actual_force = Simulate();

  // Check to within a percentage threshold. This threshold is highly dependent
  // on the force controller gains and and the simulation time.
  EXPECT_TRUE(CompareMatrices(
      Eigen::Vector2d::Ones(),
      (final_actual_force.array() / desired_force.array()).matrix(), 2e-2));
}

TEST_F(ForceControlTest, PlanarFingerStaticForceControlNegZ) {
  // Build the diagram and simulate the system.
  Vector2d desired_force(0.032, 0.065);
  BuildDiagram(BrickType::FixedBrick, Vector3d::Zero() /* brick pose */,
               desired_force, Vector2d(0.5, -1.4) /* finger q0 */,
               FingerWeldAngle(Finger::kFinger3));
  const Eigen::Vector2d final_actual_force = Simulate();

  // Check to within a percentage threshold. This threshold is highly dependent
  // on the force controller gains and and the simulation time.
  EXPECT_TRUE(CompareMatrices(
      Eigen::Vector2d::Ones(),
      (final_actual_force.array() / desired_force.array()).matrix(), 2e-2));
}

TEST_F(ForceControlTest, PlanarFingerStaticForceControlPosY) {
  // Build the diagram.
  Vector2d desired_force(-0.183, -0.119);
  BuildDiagram(BrickType::FixedBrick,
               Vector3d(-0.00854, 0.03759, 0.35880) /* brick pose */,
               desired_force, Vector2d(-0.693555, 1.18458) /* finger q0 */,
               FingerWeldAngle(Finger::kFinger2));
  const Eigen::Vector2d final_actual_force = Simulate();

  // Check to within a percentage threshold. This threshold is highly dependent
  // on the force controller gains and and the simulation time.
  EXPECT_TRUE(CompareMatrices(
      Eigen::Vector2d::Ones(),
      (final_actual_force.array() / desired_force.array()).matrix(), 2e-2));
}

TEST_F(ForceControlTest, PlanarFingerStaticForceControlPosY2) {
  // Build the diagram.
  Vector2d desired_force(-0.183, -0.020);
  BuildDiagram(BrickType::FixedBrick,
               Vector3d(-0.00854, 0.03759, -0.35880*0 + 0.11) /* brick pose */,
               desired_force, Vector2d(0.693555, -1.18458) /* finger q0 */,
               FingerWeldAngle(Finger::kFinger2));
  const Eigen::Vector2d final_actual_force = Simulate();

  // Check to within a percentage threshold. This threshold is highly dependent
  // on the force controller gains and and the simulation time.
  EXPECT_TRUE(CompareMatrices(
      Eigen::Vector2d::Ones(),
      (final_actual_force.array() / desired_force.array()).matrix(), 2e-2));
}

TEST_F(ForceControlTest, PlanarFingerStaticForceControlNegY) {
  // Build the diagram.
  Vector2d desired_force(0.183, 0.119);
  BuildDiagram(BrickType::FixedBrick,
               Vector3d(-0.00854, 0.03759, 0.35880) /* brick pose */,
               desired_force, Vector2d(-0.693555, 1.18458) /* finger q0 */,
               FingerWeldAngle(Finger::kFinger1));
  const Eigen::Vector2d final_actual_force = Simulate();

  // Check to within a percentage threshold. This threshold is highly dependent
  // on the force controller gains and and the simulation time.
  EXPECT_TRUE(CompareMatrices(
      Eigen::Vector2d::Ones(),
      (final_actual_force.array() / desired_force.array()).matrix(), 2e-2));
}

TEST_F(ForceControlTest, PlanarFingerDynamicForceControlNegY) {
  // Build the diagram.
  Vector2d desired_force(0.056, 0);
  BuildDiagram(BrickType::PlanarBrick,
               Vector3d(-0.07, -0.01, 0) /* brick pose */,
               desired_force, Vector2d(-1.5, 1.5) /* finger q0 */,
               FingerWeldAngle(Finger::kFinger1));
  const Eigen::Vector2d final_actual_force = Simulate();

  // Check to within a percentage threshold for the y-direction. This threshold
  // is highly dependent on the force controller gains and and the simulation
  // time.
  EXPECT_NEAR(1.0, final_actual_force(0) / desired_force(0), 2e-2);

  // The z-direction should be close to zero, within some arbitrary threshold.
  EXPECT_NEAR(0, final_actual_force(1), 1e-3);
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
