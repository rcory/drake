#include "drake/examples/planar_gripper/finger_brick_control.h"
#include "drake/examples/planar_gripper/finger_brick.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/constant_value_source.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/systems/lcm/connect_lcm_scope.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/examples/planar_gripper/gripper_brick.h"
#include "drake/examples/planar_gripper/brick_qp.h"
#include "drake/examples/planar_gripper/planar_finger_qp.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/multibody/plant/spatial_forces_to_lcm.h"
#include "drake/systems/primitives/zero_order_hold.h"


namespace drake {
namespace examples {
namespace planar_gripper {

using geometry::SceneGraph;
using multibody::ContactResults;
using multibody::MultibodyPlant;
using multibody::RevoluteJoint;
using systems::InputPortIndex;
using systems::InputPort;
using systems::OutputPortIndex;
using systems::OutputPort;

// Force controller with pure gravity compensation (no dynamics compensation
// yet). Regulates position in z, and force in y.
// TODO(rcory) implement dynamic compensation.

ForceController::ForceController(MultibodyPlant<double>& plant,
                                 SceneGraph<double>& scene_graph,
                                 ForceControlOptions options)
    : plant_(plant),
      scene_graph_(scene_graph),
      finger_index_(plant.GetModelInstanceByName("planar_gripper")),
      options_(options) {
  // Make context with default parameters.
  plant_context_ = plant.CreateDefaultContext();

  force_desired_input_port_ =
      this->DeclareAbstractInputPort(
              "f_d", Value<std::vector<
                         multibody::ExternallyAppliedSpatialForce<double>>>())
          .get_index();
  const int kNumFingerVelocities = 2;
  finger_state_actual_input_port_ =  // actual state of the finger (joints) (4x1 vec)
      this->DeclareVectorInputPort(
              "xa", systems::BasicVector<double>(2 * kNumFingerVelocities))
          .get_index();
  // desired state of the fingertip (y, z, ydot, zdot)
  const int kNumTipVelocities = 2;
  tip_state_desired_input_port_ =
      this->DeclareVectorInputPort(
              "tip_xd", systems::BasicVector<double>(2 * kNumTipVelocities))
          .get_index();
  contact_results_input_port_ =
      this->DeclareAbstractInputPort("contact_results",
                                     Value<ContactResults<double>>{})
          .get_index();

  const int kDebuggingOutputs = 3;
  torque_output_port_ = this->DeclareVectorOutputPort(
                                "tau",
                                systems::BasicVector<double>(
                                    plant.num_actuators() + kDebuggingOutputs),
                                &ForceController::CalcTauOutput)
                            .get_index();
}

void ForceController::CalcTauOutput(
    const systems::Context<double>& context,
    systems::BasicVector<double>* output_vector) const {
  auto output_calc = output_vector->get_mutable_value();
  output_calc.setZero();

  auto external_spatial_forces_vec =
      this->get_input_port(force_desired_input_port_)
          .Eval<std::vector<multibody::ExternallyAppliedSpatialForce<double>>>(
              context);
  DRAKE_DEMAND(external_spatial_forces_vec.size() == 1);
  Eigen::Vector3d force_des_W = /* Only care about forces in the y-z plane */
      external_spatial_forces_vec[0].F_Bq_W.translational();

  auto finger_state = /* 4-element vector of the state of the finger */
      this->EvalVectorInput(context, finger_state_actual_input_port_)->get_value();

  // TODO(rcory) Do I need this, since my desired_force input, i.e., of type
  // ExternallyAppliedSpatialForce specifies a contact point? I could
  // potentially use that as a desired tip location?
  auto tip_state_desired = /* 4-element vector of the tip state of the finger */
      this->EvalVectorInput(context, tip_state_desired_input_port_)
          ->get_value();

  // Get the actual contact force.
  const auto& contact_results =
      get_contact_results_input_port().Eval<ContactResults<double>>(context);

  Eigen::Vector3d force_sim(0, 0, 0);
  // assume only zero or one contact is possible.
  if (contact_results.num_point_pair_contacts() > 0) {
    auto contact_info = contact_results.point_pair_contact_info(0);
    force_sim = contact_info.contact_force();
  }

  // Keep only the last two components of the force (since we only care about
  // forces in the y-z plane, i.e., in the plane of motion). This force returns
  // as the force felt by the brick.
  Eigen::Vector2d force_act = force_sim.tail<2>();

  // Set the plant's position and velocity within the context.
  plant_.SetPositionsAndVelocities(plant_context_.get(), finger_index_,
                                   finger_state);

  // initialize the vector for calculated torque commands.
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
  Eigen::Matrix<double, 6, 3> Jv_V_BaseFtip(6, 3);

  // p_WC is the contact point (reference) w.r.t. the World. When in contact
  // this reference is the actual contact point (lying inside the
  // contact intersection). When not in contact this reference is the
  // fingertip sphere center.
  Eigen::Vector3d p_L2FTip = Eigen::Vector3d::Zero();
  Eigen::Vector3d p_WC = Eigen::Vector3d::Zero();
  const multibody::Frame<double>& l2_frame =
      plant_.GetBodyByName("finger_link2").body_frame();

  const multibody::Frame<double>& base_frame =
      plant_.GetBodyByName("finger_base").body_frame();

  // Desired forces (external spatial forces) are expressed in the world frame.
  // Express these in the finger base frame instead (to match the Jacobian).
  /* Rotation of world (W) w.r.t. finger base (B) */
  auto R_BW = plant_.CalcRelativeRotationMatrix(
      *plant_context_, base_frame, plant_.world_frame());
  Eigen::Vector3d force_des_base = R_BW * force_des_W;
  auto force_des = force_des_base.tail<2>();


//  drake::log()->info("force_w: \n{}", force_des_W);
//  drake::log()->info("force_base: \n{}", force_des_base);
//  drake::log()->info("rel rot: \n{}", R_BW.matrix());

  // If we have contact, then the contact point reference is given by the
  // contact results object and it lies inside the intersection of the
  // fingertip sphere and brick geometries.
  if (contact_results.num_point_pair_contacts() > 0) {
    p_WC = contact_results.point_pair_contact_info(0).contact_point();
    plant_.CalcPointsPositions(*plant_context_, plant_.world_frame(), p_WC,
                               l2_frame, &p_L2FTip);
  } else {  // otherwise we have no contact, and we take the fingertip sphere
    // center as the contact point reference.
    p_L2FTip = GetFingerTipSpherePositionInL2(plant_, scene_graph_);
    plant_.CalcPointsPositions(*plant_context_, l2_frame, p_L2FTip,
                               plant_.world_frame(), &p_WC);
  }

  plant_.CalcJacobianSpatialVelocity(
      *plant_context_, multibody::JacobianWrtVariable::kV, l2_frame, p_L2FTip,
      base_frame, base_frame, &Jv_V_BaseFtip);

  // Extract the planar translational part of the Jacobian.
  // The last two rows of Jv_V_WFtip correspond to y-z.
  // TODO(rcory) Figure out jacobian ordering.
  // Last two columns of Jacobian correspond to j1 and j2 (first column seems to
  // correspond to the brick).
  Eigen::Matrix<double, 2, 2> J(2, 2);
  J.block<1, 2>(0, 0) =
      Jv_V_BaseFtip.block<1, 2>(4, 1);
  J.block<1, 2>(1, 0) = Jv_V_BaseFtip.block<1, 2>(5, 1);

  // Force control gains (allow force control in both y and z).
  Eigen::Matrix<double, 2, 2> Kf(2, 2);
  Kf << options_.kfy_, 0, 0, options_.kfz_;

  // Regulate force in y (in world frame)
//  force_des = Eigen::Vector2d(-.1, 0);
//  drake::log()->info("force_des: \n{}", force_des);
  auto delta_f = force_des - force_act;
  // auto fy_command = Kf * delta_f + force_des;  //TODO(rcory) bug?
  auto force_error_command = Kf * delta_f;
  // auto fy_command = Kf Δf + Ki ∫ Δf dt - Kp p_e + f_d  // More general.

  // Regulate position in z (in world frame)
  // v_BFtip is translational velocity of Ftip sphere center w.r.t.
  // finger base
  auto v_BFtip = J * finger_state.segment<2>(2);  // state is for finger only
  auto delta_pos = tip_state_desired.head<2>() - p_WC.tail<2>();
  auto delta_vel = tip_state_desired.tail<2>() - v_BFtip.head<2>();
  Eigen::Matrix<double, 2, 2> Kp_pos(2, 2), Kd_pos(2, 2);
  Kp_pos << 0, 0, 0, options_.kpz_;  // position control only in z
  Kd_pos << 0, 0, 0, options_.kdz_;
  auto position_z_error_command = Kp_pos * delta_pos + Kd_pos * delta_vel;

  if (options_.always_direct_force_control_ ||
      contact_results.num_point_pair_contacts() > 0) {
    // Adds Joint damping.
    Eigen::Matrix2d Kd;
    Kd << options_.Kd_, 0, 0, options_.Kd_;
    torque_calc += -Kd * finger_state.segment<2>(2);

    // Torque due to hybrid position/force control
    torque_calc += J.transpose() *
                   (force_des + force_error_command + position_z_error_command);
  } else {
    // TODO(rcory) remove. for debugging.
    throw std::logic_error("impedance control not tested yet.");
    // regulate the fingertip position back to the surface w/ impedance control
    // implement a simple spring law for now (-kx), i.e., just compliance
    // control
    const double target_z_position = 0.1;  // arbitrary
    const double K = options_.K_compliance_;   // spring const
    const double D = options_.D_damping_;
    double z_force_desired =
        -K * (target_z_position - p_WC(2)) - D * v_BFtip(1);
    Vector2<double> imp_force_desired(0, z_force_desired);
    torque_calc += J.transpose() * imp_force_desired;
  }

  // The output for calculated torques.
  output_calc.head<2>() = torque_calc;

  // These are just auxiliary debugging outputs.
  output_calc.segment<2>(2) =
      position_z_error_command + force_error_command;
  output_calc(4) = delta_pos(0);
}

/// Creates the QP controller (type depending on whether we are simulating a
/// single brick or finger/brick), and connects it to the force controller.
void ConnectControllers(const MultibodyPlant<double>& plant,
                        lcm::DrakeLcm& lcm,
                        const ForceController& force_controller,
                        const ModelInstanceIndex& brick_index,
                        const QPControlOptions options,
                        systems::DiagramBuilder<double>* builder) {
  // thetaddot_planned is 0. Use a constant source.
  auto thetaddot_planned_source =
      builder->AddSystem<systems::ConstantVectorSource<double>>(Vector1d(0));

  // The planned theta trajectory is from 0 to 90 degree in T seconds.
  std::vector<double> T = {-0.5, 0, options.T_, options.T_ + 0.5};
  std::vector<MatrixX<double>> Y(T.size(), MatrixX<double>::Zero(1, 1));
  Y[0](0, 0) = options.theta0_;
  Y[1](0, 0) = options.theta0_;
  Y[2](0, 0) = options.thetaf_;
  Y[3](0, 0) = options.thetaf_;

//  const trajectories::PiecewisePolynomial<double> theta_traj =
//      trajectories::PiecewisePolynomial<double>::Pchip(T, Y);

  // this doesn't work well for disturbed trajectories (since the traj is only
  // computed at initialization).
//  auto theta_traj_source =
//      builder->AddSystem<systems::TrajectorySource<double>>(
//          theta_traj, 1 /* take 1st derivatives */);

  // so, just use a constant target...
  auto theta_vec = Eigen::Vector2d(options.thetaf_, 0);
  auto theta_traj_source =
      builder->AddSystem<systems::ConstantVectorSource<double>>(theta_vec);

  // Spit out to scope
  systems::lcm::ConnectLcmScope(theta_traj_source->get_output_port(),
                                "THETA_TRAJ", builder, &lcm);

  // damping for the 1-dof brick joint (strictly for the QP controller).
  double damping = 0;
  if (!options.assume_zero_brick_damping_) {
    damping = plant.GetJointByName<multibody::RevoluteJoint>("brick_pin_joint")
                  .damping();
  }

  // QP controller
  double Kp = options.QP_Kp_;
  double Kd = options.QP_Kd_;
  double weight_thetaddot_error = options.QP_weight_thetaddot_error_;
  double weight_f_Cb_B = options.QP_weight_f_Cb_B_;
  double mu = options.QP_mu_;

  // Always get in contact with the +z face.
  auto contact_face_source =
      builder->AddSystem<systems::ConstantValueSource<double>>(
          Value<BrickFace>(BrickFace::kPosZ));

  if (options.brick_only_) {
    // ================ Brick QP controller =================================
    // This implements the QP controller for brick only.
    // ======================================================================
    auto qp_controller = builder->AddSystem<BrickInstantaneousQPController>(
        &plant, Kp, Kd, weight_thetaddot_error, weight_f_Cb_B, mu, damping);

    // Connect the QP controller
    builder->Connect(plant.get_state_output_port(brick_index),
                     qp_controller->get_input_port_estimated_state());
    builder->Connect(qp_controller->get_output_port_control(),
                     plant.get_applied_spatial_force_input_port());

    // To visualize the applied spatial forces.
    auto viz_converter = builder->AddSystem<ExternalSpatialToSpatialViz>(
        plant, brick_index, -options.viz_force_scale_);
    builder->Connect(qp_controller->get_output_port_control(),
                     viz_converter->get_input_port(0));
    builder->Connect(plant.get_state_output_port(brick_index),
                     viz_converter->get_input_port(1));
    multibody::ConnectSpatialForcesToDrakeVisualizer(
        builder, plant, viz_converter->get_output_port(0), &lcm);

    builder->Connect(contact_face_source->get_output_port(0),
                     qp_controller->get_input_port_contact_face());

    // Always make contact with sphere center at position (yc, zc).
    auto p_BCb_source =
        builder->AddSystem<systems::ConstantVectorSource<double>>(
            Eigen::Vector2d(options.yc_, options.zc_));
    builder->Connect(p_BCb_source->get_output_port(),
                     qp_controller->get_input_port_p_BCb());

    builder->Connect(thetaddot_planned_source->get_output_port(),
                     qp_controller->get_input_port_desired_acceleration());
    builder->Connect(theta_traj_source->get_output_port(),
                     qp_controller->get_input_port_desired_state());
  } else {
    // ================ Planar Finger QP controller =========================
    // This implements the QP controller for brick AND planar-finger.
    // ======================================================================

    double fingertip_radius = 0.015;
    auto qp_controller =
        builder->AddSystem<PlanarFingerInstantaneousQPController>(
            &plant, Kp, Kd, weight_thetaddot_error, weight_f_Cb_B, mu,
            fingertip_radius, damping);

    // Connect the QP controller
    builder->Connect(plant.get_state_output_port(),
                     qp_controller->get_input_port_estimated_state());

    // TODO(rcory) Connect this to the force controller.
    // Note: The spatial forces coming from the output of the QP controller
    // are already in the world frame (only the contact point is in the brick
    // frame)

    builder->Connect(qp_controller->get_output_port_control(),
                     force_controller.get_force_desired_input_port());

    // To visualize the applied spatial forces.
    auto viz_converter = builder->AddSystem<ExternalSpatialToSpatialViz>(
        plant, brick_index, options.viz_force_scale_);
    builder->Connect(qp_controller->get_output_port_control(),
                     viz_converter->get_input_port(0));
    multibody::ConnectSpatialForcesToDrakeVisualizer(
        builder, plant, viz_converter->get_output_port(0), &lcm);
    builder->Connect(plant.get_state_output_port(brick_index),
                     viz_converter->get_input_port(1));

    builder->Connect(contact_face_source->get_output_port(0),
                     qp_controller->get_input_port_contact_face());

    // Adds a zero order hold to the contact results.
    auto zoh = builder->AddSystem<systems::ZeroOrderHold<double>>(
        1e-3, Value<ContactResults<double>>());
    builder->Connect(plant.get_contact_results_output_port(),
                     zoh->get_input_port());

    // Adds system to calculate fingertip contact.
    auto contact_point_calc_sys =
        builder->AddSystem<ContactPointInBrickFrame>(plant);
    builder->Connect(zoh->get_output_port(),
                     contact_point_calc_sys->get_input_port(0));
    builder->Connect(plant.get_state_output_port(),
                     contact_point_calc_sys->get_input_port(1));
    builder->Connect(contact_point_calc_sys->get_output_port(0),
                     qp_controller->get_input_port_p_BFingerTip());

    builder->Connect(thetaddot_planned_source->get_output_port(),
                     qp_controller->get_input_port_desired_thetaddot());
    builder->Connect(theta_traj_source->get_output_port(),
                     qp_controller->get_input_port_desired_state());
  }
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake