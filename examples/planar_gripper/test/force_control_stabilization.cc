#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/examples/planar_gripper/finger_brick.h"
#include "drake/examples/planar_gripper/finger_brick_control.h"
#include "drake/examples/planar_gripper/planar_gripper.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/examples/planar_gripper/planar_gripper_utils.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_value_source.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/trajectory_source.h"

namespace drake {
namespace examples {
namespace planar_gripper {
namespace {

using multibody::ContactResults;
using multibody::RevoluteJoint;

// TODO(rcory) Move all common flags to a shared YAML file.
DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_double(time_step, 1e-3,
              "If greater than zero, the plant is modeled as a system with "
              "discrete updates and period equal to this time_step. "
              "If 0, the plant is modeled as a continuous system.");
DEFINE_double(penetration_allowance, 0.2, "The contact penetration allowance.");
DEFINE_double(stiction_tolerance, 1e-3, "MBP v_stiction_tolerance");
DEFINE_double(floor_coef_static_friction, 0.5,
              "The floor's coefficient of static friction");
DEFINE_double(floor_coef_kinetic_friction, 0.5,
              "The floor's coefficient of kinetic friction");
DEFINE_double(brick_floor_penetration, 1e-4,
              "Determines how much the brick should penetrate the floor "
              "(in meters). When simulating the vertical case this penetration "
              "distance will remain fixed.");

DEFINE_double(G_ROT, 0,
              "Rotation of gripper frame (G) w.r.t. the world frame W around "
              "the x-axis (deg).");

// Hybrid position/force control parameters.
DEFINE_double(kd_base_joint, 1.0, "joint damping for base joint.");
DEFINE_double(kd_mid_joint, 1.0, "joint damping for mid joint.");
DEFINE_double(kp_t, 0, "Tangential position gain (in brick frame).");
DEFINE_double(kd_t, 20e2, "Tangential derivative gain (in brick frame).");
DEFINE_double(kp_n, 0, "Normal position gain (in brick frame).");
DEFINE_double(kd_n, 15e3, "Normal derivative gain (in brick frame).");
DEFINE_double(kpf_t, 3e3,
              "Tangential proportional force gain (in brick frame).");
DEFINE_double(kpf_n, 5e3, "Normal proportional force gain (in brick frame).");
DEFINE_double(kif_t, 1e2, "Tangential integral force gain (in brick frame).");
DEFINE_double(kif_n, 1e2, "Normal integral force gain (in brick frame).");
DEFINE_double(K_compliance, 2e3, "Impedance control stiffness.");
DEFINE_double(D_damping, 1e3, "Impedance control damping.");

DEFINE_double(viz_force_scale, 1,
              "scale factor for visualizing spatial force arrow");

// Define which fingers are used for the brick rotation.
DEFINE_bool(use_finger1, true, "Use finger1?");
DEFINE_bool(use_finger2, true, "Use finger2?");
DEFINE_bool(use_finger3, true, "Use finger3?");

// Boundary conditions.
DEFINE_double(theta0, -0.6478211916450246, "initial theta (rad)");
DEFINE_double(thetadot0, 0, "initial brick rotational velocity.");
DEFINE_double(y0, 0, "initial brick y position (m).");
DEFINE_double(z0, 0, "initial brick z position (m).");
DEFINE_double(yf, 0, "goal brick y position (m), for regulation task.");
DEFINE_double(zf, 0, "goal brick z position (m), for regulation task.");
DEFINE_double(thetaf, M_PI_4,
              "goal brick theta rotation (rad), for regulation task.");
DEFINE_bool(
    is_regulation_task, false,
    "Defines the type of control task. If set to `true`, the QP planner "
    "executes a regulation task. If set to false, the QP planner executes a "
    "tracking task. A `regulation` task controls the brick to a set-point "
    "goal. A `tracking` task controls the brick to follow a desired state "
    "trajectory.");
DEFINE_double(T, 1.5, "time horizon (s)");

// QP task parameters.
DEFINE_string(brick_type, "planar",
              "Defines the brick type: {pinned, planar}.");
DEFINE_string(keyframes_filename, "planar_brick_multi_mode.txt",
              "The name of the file containing the keyframes.");
DEFINE_double(QP_plan_dt, 0.002, "The QP planner's timestep.");
DEFINE_double(QP_Kp_t, 350, "QP controller translational Kp gain.");
DEFINE_double(QP_Kd_t, 100, "QP controller translational Kd gain.");
DEFINE_double(QP_Ki_t, 500, "QP controller translational Ki gain.");
DEFINE_double(QP_Ki_r, 19e3, "QP controller rotational Ki gain.");
DEFINE_double(QP_Ki_r_sat, 0.004, "QP integral rotational saturation value.");
DEFINE_double(QP_Ki_t_sat, 0.05,
              "QP integral translational saturation value.");
DEFINE_double(QP_Kp_r_pinned, 2e3,
              "QP controller rotational Kp gain for pinned brick.");
DEFINE_double(QP_Kd_r_pinned, 10,
              "QP controller rotational Kd gain for pinned brick.");
DEFINE_double(QP_Kp_r_planar, 195,
              "QP controller rotational Kp gain for planar brick.");
DEFINE_double(QP_Kd_r_planar, 120,
              "QP controller rotational Kd gain for planar brick.");
DEFINE_double(QP_weight_thetaddot_error, 1, "thetaddot error weight.");
DEFINE_double(QP_weight_a_error, 1, "translational acceleration error weight.");
DEFINE_double(QP_weight_f_Cb_B, 500, "Contact force magnitude penalty weight");
DEFINE_double(QP_mu, 1.0, "QP mu"); /* MBP defaults to mu1 == mu2 == 1.0 */
// TODO(rcory) Pass in QP_mu to brick and fingertip-sphere collision geoms.

DEFINE_bool(test, false,
            "If true, checks the simulation result against a known value.");
DEFINE_double(switch_time, 0.7, "Hybrid control switch time");
DEFINE_bool(print_keyframes, false,
            "Print joint positions (keyframes) to standard out?");

DEFINE_double(time_scale_factor, 0.7, "time scale factor.");
DEFINE_bool(add_floor, true, "Adds a floor to the simulation");

// For this demo, we switch from position control to torque control after a
// certain time has elapsed (given by the flag switch_time). This should occur
// right after contact is initiated with all three fingers.
// This system has a single output port that contains the control type to use
// based on the current context time.
class TimedControlType final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(TimedControlType);

  TimedControlType() {
    this->DeclareAbstractOutputPort("control_type",
                                    &TimedControlType::SetOutput);
  }
  void SetOutput(const systems::Context<double>& context,
                 ControlType* control_type) const {
    if (context.get_time() > FLAGS_switch_time) {
      *control_type = ControlType::kTorque;
    } else {
      *control_type = ControlType::kPosition;
    }
  }
};

// This system generates three outputs (one for each finger) indicating whether
// it is in position or force control mode. The system is constructed using
// a matrix of mode keyframes, which indicate which BrickFace each of the
// fingers should be in contact with (or if it should not be in contact at all).
// In this system, we ignore the desired contact BrickFace. If the finger's
// contact BrickFace is greater than -1, then we transition to force control
// mode, if the BrickFace is -1, then we transition to position control.

// TODO(rcory) Later implementations should at least monitor the "desired"
//  contact BrickFace given by the contact mode keyframes, and throw if we don't
//  match.
class KeyframeControlType final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(KeyframeControlType);

  KeyframeControlType(Eigen::VectorXd times, MatrixX<double> modes)
      : times_(times), modes_(modes) {
    this->DeclareAbstractOutputPort("finger1_control_type",
                                    &KeyframeControlType::Setf1ControlType);
    this->DeclareAbstractOutputPort("finger2_control_type",
                                    &KeyframeControlType::Setf2ControlType);
    this->DeclareAbstractOutputPort("finger3_control_type",
                                    &KeyframeControlType::Setf3ControlType);

    // Holds the current time index, initialized to zero.
    this->DeclareDiscreteState(1);
  }

  void DoCalcNextUpdateTime(
      const systems::Context<double>& context,
      systems::CompositeEventCollection<double>* events,
      double* time) const override {
    LeafSystem<double>::DoCalcNextUpdateTime(context, events, time);
    *time = times_(context.get_discrete_state(0).get_value()(0));

    // Create a discrete update event and tie the handler to the corresponding
    // function.
    systems::DiscreteUpdateEvent<double>::DiscreteUpdateCallback callback =
        [this](const systems::Context<double>& c,
               const systems::DiscreteUpdateEvent<double>& du_event,
               systems::DiscreteValues<double>* dvals) {
          this->DiscreteCallbackTest(c, du_event, dvals);
        };

    systems::EventCollection<systems::DiscreteUpdateEvent<double>>& pub_events =
        events->get_mutable_discrete_update_events();
    pub_events.add_event(std::make_unique<systems::DiscreteUpdateEvent<double>>(
        systems::TriggerType::kTimed, callback));
  }

  void Setf1ControlType(
      const systems::Context<double>& context,
      ControlType* control_type) const {
    double index = context.get_discrete_state().get_vector().get_value()(0);
    *control_type =
        (modes_(0, index) > -1) ? ControlType::kTorque : ControlType::kPosition;
  }

  void Setf2ControlType(
      const systems::Context<double>& context,
      ControlType* control_type) const {
    double index = context.get_discrete_state().get_vector().get_value()(0);
    *control_type =
        (modes_(1, index) > -1) ? ControlType::kTorque : ControlType::kPosition;
  }

  void Setf3ControlType(
      const systems::Context<double>& context,
      ControlType* control_type) const {
    double index = context.get_discrete_state().get_vector().get_value()(0);
    *control_type =
        (modes_(2, index) > -1) ? ControlType::kTorque : ControlType::kPosition;
  }

  systems::EventStatus DiscreteCallbackTest(
      const systems::Context<double>& context,
      const systems::DiscreteUpdateEvent<double>& event,
      systems::DiscreteValues<double>* values) const {
    double current_index = values->get_vector().get_value()(0);
    values->get_mutable_vector().get_mutable_value()(0) =
        std::min<double>(times_.size() - 1, current_index + 1);
    return systems::EventStatus::Succeeded();
  }

 private:
  Eigen::VectorXd times_;
  MatrixX<double> modes_;
};

class BrickPlanFrameViz final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(BrickPlanFrameViz);

  explicit BrickPlanFrameViz(int num_positions) {
    this->DeclareVectorInputPort("brick_positions",
                                 systems::BasicVector<double>(num_positions));
    this->DeclareAbstractOutputPort("brick_xform",
                                    &BrickPlanFrameViz::CalcXForm);
  }

  void CalcXForm(const systems::Context<double>& context,
                 std::vector<math::RigidTransform<double>>* xform_vec) const {
    VectorX<double> brick_q = this->EvalVectorInput(context, 0)->get_value();
    double ypos = 0, zpos = 0, theta = 0;
    if (brick_q.size() == 1) {
      theta = brick_q(0);
    } else {
      ypos = brick_q(0);
      zpos = brick_q(1);
      theta = brick_q(2);
    }
    auto xform = math::RigidTransform<double>(
        math::RollPitchYaw<double>(Vector3d(theta, 0, 0)),
        Vector3d(0, ypos, zpos));
    xform_vec->clear();
    xform_vec->push_back(xform);
  }
};

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
                         const BrickType& brick_type,
                         QPControlOptions* qpoptions) {
  double brick_rotational_damping = planar_gripper.GetBrickPinJointDamping();

  // Get the brick's Ixx moment of inertia (i.e., around the pinned axis).
  const int kIxx_index = 0;
  double brick_inertia = planar_gripper.GetBrickMoments()(kIxx_index);
  double brick_mass = planar_gripper.GetBrickMass();

  qpoptions->T_ = FLAGS_T;
  qpoptions->plan_dt = FLAGS_QP_plan_dt;
  qpoptions->brick_goal_.y_goal = FLAGS_yf;
  qpoptions->brick_goal_.z_goal = FLAGS_zf;
  qpoptions->brick_goal_.theta_goal = FLAGS_thetaf;
  qpoptions->control_task_ = (FLAGS_is_regulation_task)
                                 ? ControlTask::kRegulation
                                 : ControlTask::kTracking;
  qpoptions->QP_Kp_r_ =
      (brick_type == BrickType::PinBrick ? FLAGS_QP_Kp_r_pinned
                                         : FLAGS_QP_Kp_r_planar);
  qpoptions->QP_Kd_r_ =
      (brick_type == BrickType::PinBrick ? FLAGS_QP_Kd_r_pinned
                                         : FLAGS_QP_Kd_r_planar);
  qpoptions->QP_Ki_r_ = FLAGS_QP_Ki_r;
  qpoptions->QP_Kp_t_ =
      Eigen::Vector2d(FLAGS_QP_Kp_t, FLAGS_QP_Kp_t).asDiagonal();
  qpoptions->QP_Kd_t_ =
      Eigen::Vector2d(FLAGS_QP_Kd_t, FLAGS_QP_Kd_t).asDiagonal();
  qpoptions->QP_Ki_t_ =
      Eigen::Vector2d(FLAGS_QP_Ki_t, FLAGS_QP_Ki_t).asDiagonal();
  qpoptions->QP_Ki_r_sat_ = FLAGS_QP_Ki_r_sat;
  qpoptions->QP_Ki_t_sat_ = FLAGS_QP_Ki_t_sat;
  qpoptions->QP_weight_thetaddot_error_ = FLAGS_QP_weight_thetaddot_error;
  qpoptions->QP_weight_acceleration_error_ = FLAGS_QP_weight_a_error;
  qpoptions->QP_weight_f_Cb_B_ = FLAGS_QP_weight_f_Cb_B;
  qpoptions->QP_mu_ = FLAGS_QP_mu;
  qpoptions->brick_only_ = false;
  qpoptions->viz_force_scale_ = FLAGS_viz_force_scale;
  qpoptions->brick_rotational_damping_ = brick_rotational_damping;
  qpoptions->brick_translational_damping_ = 0;
  qpoptions->brick_inertia_ = brick_inertia;
  qpoptions->brick_mass_ = brick_mass;
  qpoptions->brick_spatial_force_assignments_ =
      BrickSpatialForceAssignments(FingersToControl());
  qpoptions->brick_type_ = brick_type;
}

void GetForceControllerOptions(const Finger finger,
                               ForceControlOptions* foptions) {
  foptions->kpf_t_ = FLAGS_kpf_t;
  foptions->kpf_n_ = FLAGS_kpf_n;
  foptions->kif_t_ = FLAGS_kif_t;
  foptions->kif_n_ = FLAGS_kif_n;
  foptions->kp_t_ = FLAGS_kp_t;
  foptions->kd_t_ = FLAGS_kd_t;
  foptions->kp_n_ = FLAGS_kp_n;
  foptions->kd_n_ = FLAGS_kd_n;
  foptions->Kd_joint_ << FLAGS_kd_base_joint, 0, 0, FLAGS_kd_mid_joint;
  foptions->K_compliance_ = FLAGS_K_compliance;
  foptions->D_damping_ = FLAGS_D_damping;
  foptions->always_direct_force_control_ = false;
  foptions->finger_to_control_ = finger;
}

int DoMain() {
  systems::DiagramBuilder<double> builder;
  auto planar_gripper = builder.AddSystem<PlanarGripper>(
      FLAGS_time_step, ControlType::kHybrid, FLAGS_add_floor);

  // Set some floor parameters.
  planar_gripper->set_floor_coef_static_friction(
      FLAGS_floor_coef_static_friction);
  planar_gripper->set_floor_coef_kinetic_friction(
      FLAGS_floor_coef_kinetic_friction);
  planar_gripper->set_brick_floor_penetration(FLAGS_brick_floor_penetration);

  // Set up the plant.
  auto X_WG = math::RigidTransformd(
      math::RollPitchYaw<double>(FLAGS_G_ROT * M_PI / 180, 0, 0),
      Eigen::Vector3d::Zero());
  planar_gripper->set_X_WG(X_WG);
  BrickType brick_type;
  if (FLAGS_brick_type == "pinned") {
    planar_gripper->SetupPinBrick("vertical");
    brick_type = BrickType::PinBrick;
  } else if (FLAGS_brick_type == "planar") {
    planar_gripper->SetupPlanarBrick("vertical");
    brick_type = BrickType::PlanarBrick;
  } else {
    throw std::runtime_error("Unknown BrickType.");
  }
  planar_gripper->set_penetration_allowance(FLAGS_penetration_allowance);
  planar_gripper->set_stiction_tolerance(FLAGS_stiction_tolerance);
  planar_gripper->zero_gravity();

  // Finalize and build the diagram.
  planar_gripper->Finalize();

  lcm::DrakeLcm drake_lcm;

  QPControlOptions qpoptions;
  GetQPPlannerOptions(*planar_gripper, brick_type, &qpoptions);

  std::unordered_map<Finger, ForceController&> finger_force_control_map;
  for (auto& finger : FingersToControl()) {
    ForceControlOptions foptions;
    GetForceControllerOptions(finger, &foptions);
    DRAKE_DEMAND(finger == foptions.finger_to_control_);
    ForceController* force_controller =
        SetupForceController(*planar_gripper, &drake_lcm, foptions, &builder);
    finger_force_control_map.emplace(finger, *force_controller);
  }

  // Parse the keyframes
  const std::string keyframe_path =
      "drake/examples/planar_gripper/keyframes/" + FLAGS_keyframes_filename;
  MatrixX<double> finger_keyframes;
  std::map<std::string, int> finger_joint_name_to_row_index_map;
  std::pair<MatrixX<double>, std::map<std::string, int>> brick_keyframe_info;

  VectorX<double> times;
  MatrixX<double> modes;
  std::tie(finger_keyframes, finger_joint_name_to_row_index_map) =
      ParseKeyframesAndModes(keyframe_path, &times, &modes,
                             &brick_keyframe_info);
  DRAKE_DEMAND(times.size() == finger_keyframes.cols());
  DRAKE_DEMAND(modes.rows() == 3 && modes.cols() == finger_keyframes.cols());
  finger_keyframes = ReorderKeyframesForPlant(
      planar_gripper->get_control_plant(), finger_keyframes,
      &finger_joint_name_to_row_index_map);

  // time rescaling hack.
  times = times * FLAGS_time_scale_factor;

  MatrixX<double> brick_keyframes = brick_keyframe_info.first;
  std::map<std::string, int> brick_joint_name_to_row_index_map =
      brick_keyframe_info.second;

  // Create the individual finger matrices. For this demo, we assume we have
  // three fingers, two joints each.
  DRAKE_DEMAND(kNumFingers == 3 && kNumJointsPerFinger == 2);
  int num_keys = finger_keyframes.cols();

  // Create the brick's desired state trajectory, created from a
  // piece-wise polynomial of the brick position keyframes. The brick's
  // trajectory source is created/connected in ConnectQPController().
  if (FLAGS_brick_type == "pinned") {
    qpoptions.desired_brick_traj_ =
        trajectories::PiecewisePolynomial<double>::CubicShapePreserving(
            times,
            brick_keyframes.row(
                brick_joint_name_to_row_index_map["brick_revolute_x_joint"]));
  } else {  // brick_type is planar
    MatrixX<double> ordered_brick_keyframes = brick_keyframes;
    ordered_brick_keyframes.row(0) = brick_keyframes.row(
        brick_joint_name_to_row_index_map["brick_translate_y_joint"]);
    ordered_brick_keyframes.row(1) = brick_keyframes.row(
        brick_joint_name_to_row_index_map["brick_translate_z_joint"]);
    ordered_brick_keyframes.row(2) = brick_keyframes.row(
        brick_joint_name_to_row_index_map["brick_revolute_x_joint"]);
    qpoptions.desired_brick_traj_ =
        trajectories::PiecewisePolynomial<double>::CubicShapePreserving(
            times, ordered_brick_keyframes, true);
  }

  // Create and connect the finger trajectory sources, created from a piece-wise
  // polynomial of the finger position keyframes.
  for (int i = 0; i < kNumFingers; i++) {
    MatrixX<double> fn_keyframes(2, num_keys);
    fn_keyframes.row(0) = finger_keyframes.row(
        finger_joint_name_to_row_index_map["finger" + std::to_string(i + 1) +
                                           "_BaseJoint"]);
    fn_keyframes.row(1) = finger_keyframes.row(
        finger_joint_name_to_row_index_map["finger" + std::to_string(i + 1) +
                                           "_MidJoint"]);
    const auto finger_pp =
        trajectories::PiecewisePolynomial<double>::CubicShapePreserving(
            times, fn_keyframes);
    const auto finger_state_src =
        builder.AddSystem<systems::TrajectorySource<double>>(
            finger_pp, 1 /* with one derivative */);
    builder.Connect(finger_state_src->get_output_port(),
                    planar_gripper->GetInputPort(
                        "finger" + std::to_string(i + 1) + "_desired_state"));
  }

  auto control_type_src = builder.AddSystem<KeyframeControlType>(times, modes);

  // Add and connect the control_type source system.
//  auto control_type_src = builder.AddSystem<TimedControlType>();
  DRAKE_DEMAND(finger_force_control_map.size() == kNumFingers);
  for (auto iter = finger_force_control_map.begin();
       iter != finger_force_control_map.end(); ++iter) {
    Finger finger = iter->first;
    ForceController& force_controller = finger_force_control_map.at(finger);
    builder.Connect(
        force_controller.get_torque_output_port(),
        planar_gripper->GetInputPort(to_string(finger) + "_torque_control_u"));
    builder.Connect(
        control_type_src->GetOutputPort(to_string(finger) + "_control_type"),
        planar_gripper->GetInputPort(to_string(finger) + "_control_type"));
    }

    ConnectQPController(*planar_gripper, &drake_lcm, finger_force_control_map,
                        qpoptions, &builder);

    // publish body frames.
    auto frame_viz_bodies = builder.AddSystem<FrameViz>(
        planar_gripper->get_multibody_plant(), &drake_lcm, 1.0 / 60.0, false);
    builder.Connect(planar_gripper->GetOutputPort("plant_state"),
                    frame_viz_bodies->get_input_port(0));

    auto frame_viz_planned_path = builder.AddSystem<FrameViz>(
        planar_gripper->get_multibody_plant(), &drake_lcm, 1.0 / 60.0, true);
  auto brick_desired_pos_traj_source =
      builder.AddSystem<systems::TrajectorySource<double>>(
          qpoptions.desired_brick_traj_);
  builder.Connect(planar_gripper->GetOutputPort("plant_state"),
                  frame_viz_planned_path->get_input_port(0));
  auto brick_xform_src = builder.AddSystem<BrickPlanFrameViz>(
      planar_gripper->get_num_brick_positions());
  builder.Connect(brick_desired_pos_traj_source->get_output_port(),
                  brick_xform_src->get_input_port(0));
  builder.Connect(brick_xform_src->get_output_port(0),
                  frame_viz_planned_path->get_input_port(1));

//  math::RigidTransformd goal_frame;
//  goal_frame.set_rotation(math::RollPitchYaw<double>(FLAGS_thetaf, 0, 0));
//  PublishFramesToLcm("GOAL_FRAME", {goal_frame}, {"goal"}, &drake_lcm);

  // Connect drake visualizer.
  geometry::ConnectDrakeVisualizer(
      &builder, planar_gripper->get_mutable_scene_graph(),
      planar_gripper->GetOutputPort("pose_bundle"), &drake_lcm);

  // Publish contact results for visualization.
  ConnectContactResultsToDrakeVisualizer(
      &builder, planar_gripper->get_mutable_multibody_plant(),
      planar_gripper->GetOutputPort("contact_results"));

  if (FLAGS_print_keyframes) {
    // Prints the state to standard output.
    auto joint_ordering = GetPreferredGripperJointOrdering();
    joint_ordering.emplace_back("brick_revolute_x_joint");
    auto state_pub = builder.AddSystem<PrintKeyframes>(
        planar_gripper->get_multibody_plant(), joint_ordering, 0.1,
        false /* don't print time */);
    builder.Connect(planar_gripper->GetOutputPort("plant_state"),
                    state_pub->GetInputPort("plant_state"));
    }

    auto diagram = builder.Build();

  // Create the initial conditions.
  VectorX<double> gripper_initial_positions =
      finger_keyframes.block(0, 0, kNumGripperJoints, 1);

  // Create a context for the diagram.
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& planar_gripper_context =
      diagram->GetMutableSubsystemContext(*planar_gripper,
                                          diagram_context.get());

  planar_gripper->SetGripperPosition(&planar_gripper_context,
                                     gripper_initial_positions);
  planar_gripper->SetGripperVelocity(&planar_gripper_context,
                                     VectorX<double>::Zero(kNumGripperJoints));

  std::map<std::string, double> init_brick_pos_map;
  const int rx_index = brick_keyframe_info.second["brick_revolute_x_joint"];
  init_brick_pos_map["brick_revolute_x_joint"] =
      brick_keyframe_info.first(rx_index, 0);
  if (FLAGS_brick_type == "planar") {
    const int ty_index = brick_keyframe_info.second["brick_translate_y_joint"];
    const int tz_index = brick_keyframe_info.second["brick_translate_z_joint"];
    init_brick_pos_map["brick_translate_y_joint"] =
        brick_keyframe_info.first(ty_index, 0);
    init_brick_pos_map["brick_translate_z_joint"] =
        brick_keyframe_info.first(tz_index, 0);
  }
  auto brick_initial_positions =
      planar_gripper->MakeBrickPositionVector(init_brick_pos_map);
  planar_gripper->SetBrickPosition(&planar_gripper_context,
                                   brick_initial_positions);

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(times.tail(1)(0));

  // TODO(rcory) Implement a proper unit test once all shared parameters are
  //  moved to a YAML file.
  if (FLAGS_test) {
    VectorX<double> x_known(18);
    x_known << -0.1084140, 0.8320280, -0.9261075, 0.0103368, -0.3157280,
        -1.3742861, 1.1213269, -0.0037305, 1.5645424, -0.0488483, 0.0323931,
        -0.0975373, 0.0012728, 0.1022895, 0.0680081, 0.1914062, -0.0044967,
        -0.0784012;
    const auto& post_sim_context = simulator.get_context();
    const auto& post_plant_context = diagram->GetSubsystemContext(
        planar_gripper->get_mutable_multibody_plant(), post_sim_context);
    const auto post_plant_state =
        planar_gripper->get_multibody_plant().GetPositionsAndVelocities(
            post_plant_context);

    // Check to within an arbitrary threshold.
    DRAKE_DEMAND(x_known.isApprox(post_plant_state, 1e-5));
  }

  return 0;
}

}  // namespace
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("A simple planar gripper example.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::planar_gripper::DoMain();
}
