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
#include "drake/examples/planar_gripper/planar_finger_qp.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/examples/planar_gripper/contact_force_qp.h"
#include "drake/multibody/plant/spatial_forces_to_lcm.h"
#include "drake/systems/primitives/zero_order_hold.h"
#include "drake/systems/primitives/discrete_derivative.h"
#include "Eigen/src/Core/Matrix.h"
#include "drake/examples/planar_gripper/planar_gripper.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/lcm/drake_lcm.h"

#include <cmath>

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
// yet).
// TODO(rcory) implement dynamic compensation.
ForceController::ForceController(const MultibodyPlant<double>& plant,
                                 const SceneGraph<double>& scene_graph,
                                 ForceControlOptions options,
                                 ModelInstanceIndex gripper_index,
                                 ModelInstanceIndex brick_index)
    : plant_(plant),
      scene_graph_(scene_graph),
      gripper_index_(gripper_index),
      brick_index_(brick_index),
      options_(options) {
  // Make context with default parameters.
  plant_context_ = plant.CreateDefaultContext();

  force_desired_input_port_ =
      this->DeclareAbstractInputPort(
              "f_d", Value<std::vector<
                         multibody::ExternallyAppliedSpatialForce<double>>>())
          .get_index();
  finger_state_actual_input_port_ =  // actual state of the finger (joints) (4x1 vec)
      this->DeclareVectorInputPort(
              "finger_x_act", systems::BasicVector<double>(4))
          .get_index();
  gripper_state_actual_input_port_ =  // actual state of the entire plant
      this->DeclareVectorInputPort(
          "gripper_x_act",
          systems::BasicVector<double>(plant.num_positions(gripper_index) +
                                       plant.num_velocities(gripper_index)))
  .get_index();
  brick_state_actual_input_port_ =  // actual state of the brick (joint) (2x1 vec)
      this->DeclareVectorInputPort(
              "brick_x_act", systems::BasicVector<double>(2))
          .get_index();
  // desired state of the fingertip (x, y, z, xdot, ydot, zdot)
  // we ignore the x-components in the controller.
  // TODO(rcory) Not sure how I should treat this input, since we're only
  //  regulating normal fingertip velocity in this example.
  tip_state_desired_input_port_ =
      this->DeclareVectorInputPort(
              "tip_xd", systems::BasicVector<double>(6))
          .get_index();

  // Contact point reference acceleration (for ID control).
  // TODO(rcory) likely don't need this, consider removing this input port.
  contact_point_ref_accel_input_port_ = this->DeclareVectorInputPort(
      "contact point ref accel", systems::BasicVector<double>(2)).get_index();

  contact_results_input_port_ =
      this->DeclareAbstractInputPort("contact_results",
                                     Value<ContactResults<double>>{})
          .get_index();

  // This force sensor contains {x, y, z} forces in the world (W) frame.
  force_sensor_input_port_ =
      this->DeclareVectorInputPort("force_sensor_wrench",
                                   systems::BasicVector<double>(3))
          .get_index();

  // TODO(rcory) likely don't need this, consider removing this input port.
  accelerations_actual_input_port_ =  // actual accelerations of the MBP (num_vel x 1)
      this->DeclareVectorInputPort("plant_vdot", systems::BasicVector<double>(
                                                     plant_.num_velocities()))
          .get_index();

  const int kNumTorques = 2; /* This only works for a single finger for now */
  torque_output_port_ =
      this->DeclareVectorOutputPort(
              "tau", systems::BasicVector<double>(kNumTorques),
              &ForceController::CalcTauOutput)
          .get_index();

  p_BrFingerTip_input_port_ =
      this->DeclareInputPort(systems::kVectorValued, 2).get_index();

  is_contact_input_port_ =
      this->DeclareAbstractInputPort("is_contact",
                                     Value<bool>{}).get_index();

  this->DeclareContinuousState(3);  // stores ∫ Δf dt
}

void ForceController::DoCalcTimeDerivatives(
    const drake::systems::Context<double>& context,
    drake::systems::ContinuousState<double>* derivatives) const {
  auto external_spatial_forces_vec =
      get_force_desired_input_port()
          .Eval<std::vector<multibody::ExternallyAppliedSpatialForce<double>>>(
              context);
  DRAKE_DEMAND(external_spatial_forces_vec.size() == 1);
  Eigen::Vector3d force_des_W = /* Note: we only care about forces in the y-z plane */
      external_spatial_forces_vec[0].F_Bq_W.translational();

  Eigen::Vector3d force_sensor_vec_W =
      this->EvalVectorInput(context, force_sensor_input_port_)->get_value();

  // The derivative of the continuous state is the instantaneous force error (in
  // world frame, W). We convert this to brick frame (Br) in CalcTauOutput.
  systems::VectorBase<double>& derivatives_vector =
      derivatives->get_mutable_vector();
  // Check whether there is contact between the fingertip and the brick.
  const bool& is_contact = get_is_contact_input_port().Eval<bool>(context);
  Vector3d controlled_force_diff;
  if (is_contact) {
    controlled_force_diff = force_des_W - force_sensor_vec_W;
  } else {
    // Intergral error, which is stored in the continuous state.
    // Drive the integral error to zero when there is no contact (i.e., reset).
    const systems::VectorBase<double>& state_vector =
        context.get_continuous_state_vector();
    const Eigen::VectorBlock<const VectorX<double>> state_block =
        dynamic_cast<const systems::BasicVector<double>&>(state_vector)
            .get_value();
    const int kIntegralLeakGain = 10.0;
    controlled_force_diff = -kIntegralLeakGain * state_block;
  }
  derivatives_vector.SetFromVector(controlled_force_diff);
}

void ForceController::CalcTauOutput(
    const systems::Context<double>& context,
    systems::BasicVector<double>* output_vector) const {
  auto output_calc = output_vector->get_mutable_value();
  output_calc.setZero();

  // The finger to control.
  std::string fnum = to_string(options_.finger_to_control_);

  // Run through the port evaluations.
  auto external_spatial_forces_vec =
      get_force_desired_input_port()
          .Eval<std::vector<multibody::ExternallyAppliedSpatialForce<double>>>(
              context);
  DRAKE_DEMAND(external_spatial_forces_vec.size() == 1);
  Eigen::Vector3d force_des_W = /* Note: we only care about forces in the y-z plane */
      external_spatial_forces_vec[0].F_Bq_W.translational();

  VectorX<double> finger_state(4);
  finger_state = /* 4-element vector of the state of the finger */
      this->EvalVectorInput(context, finger_state_actual_input_port_)
          ->get_value();

  auto gripper_state = /* gripper state (all fingers) */
      this->EvalVectorInput(context, gripper_state_actual_input_port_)
          ->get_value();

  Eigen::Vector2d brick_state = /* 2-element vector of the state of the brick */
      this->EvalVectorInput(context, brick_state_actual_input_port_)->get_value();

  // TODO(rcory) Do I need this, since my desired_force input, i.e., of type
  // ExternallyAppliedSpatialForce specifies a contact point? I could
  // potentially use that as a desired tip location?

  // 6-element vector of the desired contact-point-reference state in the brick
  // frame (x_des, y_des, z_des, xdot_des, ydot_des, zdot_des).
  // Controller ignores x-component.
  Vector6<double> contact_ref_state_desired_Br =
      this->EvalVectorInput(context, tip_state_desired_input_port_)
          ->get_value();

  // Get the contact point reference translational accelerations (yddot, zddot),
  // for inverse dynamics control.
  auto a_BrCr = /* 2-element vector of the accels of the contact point ref. */
      this->EvalVectorInput(context, contact_point_ref_accel_input_port_)
          ->get_value();
  unused(a_BrCr);

  // TODO(rcory) Remove this input port once contact point estimation exists.
  const auto& contact_results =
      get_contact_results_input_port().Eval<ContactResults<double>>(context);

  Eigen::Vector3d force_sensor_vec_W =
      this->EvalVectorInput(context, force_sensor_input_port_)->get_value();

  // Get the plant vdot for dynamics inverse dynamics.
  // TODO(rcory) I don't need this input anymore(?). Consider removing.
  auto plant_vdot =
      this->EvalVectorInput(context, accelerations_actual_input_port_)
          ->get_value();
  unused(plant_vdot);

  // Set the plant's position and velocity within the context.
  plant_.SetPositionsAndVelocities(plant_context_.get(), gripper_index_,
                                   gripper_state);
  plant_.SetPositionsAndVelocities(plant_context_.get(), brick_index_,
                                   brick_state);

  // Define some important frames.
  const multibody::Frame<double>& tip_link_frame =
      plant_.GetBodyByName(fnum + "_tip_link").body_frame();

  const multibody::Frame<double>& base_frame =
      plant_.GetBodyByName(fnum + "_base").body_frame();

  const multibody::Frame<double>& brick_frame =
      plant_.GetBodyByName("brick_link").body_frame();

  /* Rotation of world (W) w.r.t. brick (Br) */
  auto R_BrW = plant_.CalcRelativeRotationMatrix(
      *plant_context_, brick_frame, plant_.world_frame());

  /* Rotation of brick frame (Br) w.r.t. finger base frame (Ba) */
  auto R_BaBr = plant_.CalcRelativeRotationMatrix(
      *plant_context_, base_frame, brick_frame);

  /* Rotation of finger base frame (Ba) w.r.t. brick frame (Br) */
  auto R_BrBa = plant_.CalcRelativeRotationMatrix(
      *plant_context_, brick_frame, base_frame);

  // Initialize the vector for calculated torque commands.
  Eigen::Vector2d torque_calc(0, 0);

  // Gravity compensation.
  // TODO(rcory) why is this not the first two components of
  //  CalcGravityGeneralizedForces? Joint ordering indicates this should be
  //  the case...
   torque_calc =
       -plant_.CalcGravityGeneralizedForces(*plant_context_).segment<2>(1);

  // p_WC is the contact point (reference) w.r.t. the World. When in contact
  // this reference is the actual contact point (lying inside the
  // contact intersection). When not in contact this reference is the
  // fingertip sphere center.
  Eigen::Vector3d p_LtFTip = Eigen::Vector3d::Zero();  /* ftip in tip link */
  Eigen::Vector3d p_WC = Eigen::Vector3d::Zero();

  // Check whether there is contact between the fingertip and the brick.
  const bool& is_contact = get_is_contact_input_port().Eval<bool>(context);

  // If we have contact, then the contact point reference is given by the
  // contact results object and it lies inside the intersection of the
  // fingertip sphere and brick geometries.
  if (is_contact) {
    p_WC = contact_results.point_pair_contact_info(0).contact_point();
    plant_.CalcPointsPositions(*plant_context_, plant_.world_frame(), p_WC,
                               tip_link_frame, &p_LtFTip);
  } else {  // otherwise we have no contact, and we take the fingertip sphere
    // center as the contact point reference.
    p_LtFTip = GetFingerTipSpherePositionInLt(plant_, scene_graph_,
                                              options_.finger_to_control_);
    plant_.CalcPointsPositions(*plant_context_, tip_link_frame, p_LtFTip,
                               plant_.world_frame(), &p_WC);
  }

  // Compute the Jacobian.
  // For the finger/1-dof brick case, the plant consists of 3 dofs total (2 of
  // which belong to the finger). The resultant Jacobian will be of size 6 x 3.
  // This is the Jacobian of the finger tip contact reference (lies in the
  // intersection of geometry when there is contact), w.r.t. the finger base
  // frame. When there is no contact, it is the fingertip sphere center, w.r.t
  // the base frame.
  MatrixX<double> Jv_V_BaseFtip(6, plant_.num_velocities());

  plant_.CalcJacobianSpatialVelocity(
      *plant_context_, multibody::JacobianWrtVariable::kV, tip_link_frame,
      p_LtFTip, base_frame, base_frame, &Jv_V_BaseFtip);

  // Extract the 6 x 2 Jacobian that corresponds to finger only (ignore the
  // brick).
  int j1_index = static_cast<int>(
      plant_.GetJointByName(fnum + "_BaseJoint").velocity_start());
  int j2_index = static_cast<int>(
      plant_.GetJointByName(fnum + "_MidJoint").velocity_start());
  Eigen::Matrix<double, 6, 2> Jtemp(6, 2);
  Jtemp.block<6, 1>(0, 0) = Jv_V_BaseFtip.block<6, 1>(0, j1_index);
  Jtemp.block<6, 1>(0, 1) = Jv_V_BaseFtip.block<6, 1>(0, j2_index);

  // Extract the translational part of the Jacobian.
  // The last three rows of Jv_V_WFtip correspond to x-y-z.
  // Two columns of Jacobian correspond to j1 and j2.
  Eigen::Matrix<double, 3, 2> J_Ba(3, 2);
  J_Ba.block<1, 2>(0, 0) = Jtemp.block<1, 2>(3, 0);
  J_Ba.block<1, 2>(1, 0) = Jtemp.block<1, 2>(4, 0);
  J_Ba.block<1, 2>(2, 0) = Jtemp.block<1, 2>(5, 0);

  // Regulate position (in brick frame). First, rotate the contact point
  // reference into the brick frame.
  Eigen::Vector3d p_BrC = R_BrW * p_WC;

  // Extract the planar only (y-z) translational jacobian from the 3D (x,y,z)
  // translational jacobian.
  Eigen::Matrix<double, 2, 2> J_planar_Ba = J_Ba.block<2, 2>(1, 0);

  // v_Ftip_Ba is the translational velocity of the fingertip contact reference
  // w.r.t. the finger base (Ba)
  Eigen::Vector3d v_Ftip_Ba = J_Ba * finger_state.segment<2>(2);
  // Convert fingertip translation velocity to brick frame.
  Eigen::Vector3d v_Ftip_Br = R_BrBa * v_Ftip_Ba;

  // The contact reference translational acceleration, in the finger base frame.
  Eigen::Vector3d a_BrCr_Ba = Eigen::Vector3d::Zero();
  unused(a_BrCr_Ba);

//#define ADD_ACCELS
//#define SKIP_FORCE_CONT
#ifdef SKIP_FORCE_CONT
  unused(v_Ftip_Br);
  unused(p_BrC);
  unused(R_BaBr);
  unused(contact_ref_state_desired_Br);
  unused(force_des_W);

  // TODO(rcory) Remove when done.
  // For testing, follow a sine wave traj on the y-axis;
  /* Rotation of world (W) w.r.t. finger brick (Br) */
  auto X_BaW = plant_.CalcRelativeTransform(
      *plant_context_, base_frame, plant_.world_frame());
  //  unused(R_BaW);
  //  drake::log()->info("X_BaW: \n{}", X_BaW.matrix());

  Eigen::Vector3d p_BaC = X_BaW * p_WC;

  //  drake::log()->info("p_WC: \n{}", p_WC); // y=0.049145; z=0.0704776
  //  drake::log()->info("p_BaC: \n{}", p_BaC); // y=0.049145; z=-0.119522
  //  drake::log()->info("time: {}", context.get_time());

  double a_Kp = 25; double a_Kd = 5;
  double yoff_Ba = 0.049145; double zoff_Ba = -0.119522;

  double A = 0.025; double w = 3 * (2 * 3.14);
  double t = context.get_time();
  double yddot_Ba = (-A * w * w * sin(w * t))
      + a_Kd * ((A * w * cos(w * t)) - v_Ftip_Ba(1))
      + a_Kp * ((A * sin(w * t) + yoff_Ba) - p_BaC(1));
  double zddot_Ba = a_Kd * (0 - v_Ftip_Ba(2)) + a_Kp * (zoff_Ba - p_BaC(2));

  //  double yddot_Ba = a_Kp * (yoff_Ba - 0.03 - p_BaC(1)) + a_Kd * (0 - v_Ftip_Ba(1));
  //  double zddot_Ba = a_Kp * (zoff_Ba + 0.01 - p_BaC(2)) + a_Kd * (0 - v_Ftip_Ba(2));

  a_BrCr_Ba << 0, yddot_Ba, zddot_Ba;

//  drake::log()->info("a_BrCr_Ba: \n{}", a_BrCr_Ba);
#else
  // First case: direct force control.
  if (options_.always_direct_force_control_ || is_contact) {
    drake::log()->info("In direct force control.");
    // Desired forces (external spatial forces) are expressed in the world frame.
    // Express these in the brick frame instead (to compute the force error
    // terms), we then convert these commands to the finger base frame to match
    // the Jacobian.
    Eigen::Vector3d force_des_Br = R_BrW * force_des_W;

    // Force control gains. Allow force control in both (brick frame) y and z.
    Eigen::Matrix<double, 3, 3> Kf(3, 3);
    Kf << 0, 0, 0,  /* don't care about x */
        0, options_.kpfy_, 0,
        0, 0, options_.kpfz_;

    // Rotate the force reported by simulation to brick frame.
    Eigen::Vector3d force_act_Br = R_BrW * force_sensor_vec_W;

    // Regulate force (in brick frame)
    Eigen::Vector3d delta_f_Br = force_des_Br - force_act_Br;
    // auto fy_command = Kf * delta_f + force_des;  //TODO(rcory) bug?
    Eigen::Vector3d force_error_command_Br = Kf * delta_f_Br;

    // TODO(rcory) More general.
    // auto fy_command = Kf Δf + Ki ∫ Δf dt - Kp p_e + f_d

    // Intergral error, which is stored in the continuous state.
    const systems::VectorBase<double>& state_vector =
        context.get_continuous_state_vector();
    const Eigen::VectorBlock<const VectorX<double>> state_block =
        dynamic_cast<const systems::BasicVector<double>&>(state_vector)
            .get_value();

    Eigen::Matrix<double, 3, 3> Ki(3, 3);
    Ki << 0, 0, 0,  /* don't care about x */
        0, options_.kify_, 0,
        0, 0, options_.kifz_;
    Eigen::Vector3d force_error_integral_Br = R_BrW * state_block;
    Eigen::Vector3d force_integral_command_Br = Ki * force_error_integral_Br;

    // Compute the errors.
    Eigen::Vector3d delta_pos_Br =
        contact_ref_state_desired_Br.head<3>() - p_BrC.tail<3>();
    Eigen::Vector3d delta_vel_Br =
        contact_ref_state_desired_Br.tail<3>() - v_Ftip_Br.head<3>();
    Eigen::Matrix<double, 3, 3> Kp_pos(3, 3), Kd_pos(3, 3);
    Kp_pos.setZero();  // don't regulate position.
    Kd_pos << 0, 0, 0, 0, options_.kdy_, 0, 0, 0, options_.kdz_;  // regulate velocity in z (brick frame)
    Vector3d position_error_command_Br = Kp_pos * delta_pos_Br + Kd_pos * delta_vel_Br;

// ============ Begin inverse dynamics calcs. ===============
#ifdef ADD_ACCELS
    // Use the planned externally applied spatial force to compute a_BrCr_Ba,
    // i.e., the planned acceleration of point Bq (the contact point ref) in the
    // planar gripper base frame (Ba). This calculation depends on the brick
    // dynamics.
    Eigen::Vector3d p_BrCb = external_spatial_forces_vec[0]
                                .p_BoBq_B; /* contact point in brick frame */

    // Option 1: The angular accel. of the brick due to *planned* contact force.
    double thetaddot =
        (p_BrCb(1) * force_des_Br(2) - p_BrCb(2) * force_des_Br(1) -
         options_.brick_damping_ * brick_state(1)) /
        options_.brick_inertia_;

    //    // Option 2: Instead use the actual accel of the brick;
    //    int brick_joint_vel_index =
    //        plant_.GetJointByName("brick_pin_joint", brick_index_).velocity_start();
    //    double thetaddot = plant_vdot[brick_joint_vel_index];

    // Now calculate the desired reference accels (a_BrCr_Ba)
    // ac==centrepital, at==tangential, normalized accel. vectors.
    Eigen::Vector3d ac_unit_Ba = -(R_BaBr * p_BrCb.normalized());
    Eigen::Vector3d at_unit_Ba =
        Eigen::Vector3d(0, -p_BrCb(2), p_BrCb(1)).normalized();
    a_BrCr_Ba = (ac_unit_Ba * brick_state(1) * brick_state(1)) +
                (at_unit_Ba * p_BrCb.norm() * thetaddot);
#endif
// ============ End of inverse dynamics calcs. ===============

    // Adds Joint damping.
    torque_calc += -options_.Kd_ * finger_state.segment<2>(2);

    drake::log()->info("force_des_Br: \n{}", force_des_Br);
    drake::log()->info("force_actual_Br: \n{}", force_act_Br);
    drake::log()->info("force_delta_Br: \n{}", delta_f_Br);
    drake::log()->info("force_error_command_Br: \n{}", force_error_command_Br);
    drake::log()->info("force_integral_command_Br: \n{}", force_integral_command_Br);

    // Torque due to hybrid position/force control
    Eigen::Vector3d force_command_Ba = (R_BaBr * force_des_Br) +
                                       (R_BaBr * force_error_command_Br) +
                                       (R_BaBr * position_error_command_Br) +
                                       (R_BaBr * force_integral_command_Br);

#ifdef MY_DEBUG
    drake::log()->info("force_des_W: \n{}", force_des_W);
    drake::log()->info("force_des_Br: \n{}", force_des_Br);
    drake::log()->info("force_act_Br: \n{}", force_act_Br);
    drake::log()->info("force_error_command_Br: \n{}", force_error_command_Br);
    drake::log()->info("position_error_command_Br: \n{}", position_error_command_Br);
    drake::log()->info("R_BaBr: \n{}", R_BaBr.matrix());
    drake::log()->info("J_planar_Ba: \n{}", J_planar_Ba);
    drake::log()->info("force_command_Ba: \n{}", force_command_Ba);
#endif

    torque_calc += J_planar_Ba.transpose() * force_command_Ba.tail<2>();

    // TODO(rcory) only for debugging. Fixed desired force.
    // unused(force_command_Ba);
    // torque_calc += J_planar_Ba.transpose() * Eigen::Vector2d(0, -50);
  } else {  // Second Case: impedance control back to the brick's surface.
    // First, obtain the closest point on the brick from the fingertip sphere.
    drake::log()->info("In impedance force control.");
    Eigen::Vector2d target_position_Br =
        get_p_BrFingerTip_input_port().Eval(context);

//    drake::log()->info("finger_state: \n{}", finger_state);
//    drake::log()->info("brick_state: \n{}", brick_state);
//    drake::log()->info("J_planar_Ba: \n{}", J_planar_Ba);
//    drake::log()->info("p_BrFingerTip: \n{}", target_position_Br);
//    drake::log()->info("p_BrC: \n{}", p_BrC);
//    drake::log()->info("v_Ftip_Ba: \n{}", v_Ftip_Ba);
//    drake::log()->info("v_Ftip_Br: \n{}", v_Ftip_Br);

    // Regulate the fingertip position back to the surface w/ impedance control
    // implement a simple spring law for now (-kx), i.e., just compliance
    // control
    // const double target_z_position_Br = 0.056 - .001;  // arbitrary, in brick frame
    const double K = options_.K_compliance_; // spring const
    const double D = options_.D_damping_; // damper
    double y_force_desired_Br =
        K * (target_position_Br[0] - p_BrC(1)) - D * v_Ftip_Br(1);
    double z_force_desired_Br =
        K * (target_position_Br[1] - p_BrC(2)) - D * v_Ftip_Br(2);
    Vector3<double> imp_force_desired_Br(0, y_force_desired_Br,
                                         z_force_desired_Br);

    Vector3<double> imp_force_desired_Ba = R_BaBr * imp_force_desired_Br;
    torque_calc += J_planar_Ba.transpose() * imp_force_desired_Ba.tail(2);

    // TODO(rcory) Need to calculate a_BrCr_Ba for impedance control part.

//    drake::log()->info("imp_force_desired_Br: \n{}", imp_force_desired_Br);
//    drake::log()->info("imp_force_desired_Ba: \n{}", imp_force_desired_Ba);
  }
#endif

#ifdef ADD_ACCELS
  // Compute the reference acceleration vector.
  Eigen::Vector3d a_bias_WCr_Ba =
      plant_
          .CalcBiasForJacobianTranslationalVelocity(
              *plant_context_, multibody::JacobianWrtVariable::kV,
              tip_link_frame, p_LtFTip, plant_.world_frame(), base_frame)
          .tail<3>();
  Eigen::Vector2d qddot_finger =
      J_planar_Ba.inverse() * (a_BrCr_Ba.tail<2>() - a_bias_WCr_Ba.tail<2>());
  Eigen::Vector3d vdot_ref = Eigen::Vector3d::Zero();
  vdot_ref.tail<2>() = qddot_finger;
  multibody::MultibodyForces<double> external_forces(plant_);
  //  plant_.CalcForceElementsContribution(*plant_context_, &external_forces);
  // Eigen::Vector3d vdot_ref(0, 0, 0); /* accels for (brick, j1, j2) */
  Eigen::Vector3d id_torques =  plant_.CalcInverseDynamics(
      *plant_context_, vdot_ref, external_forces);
  torque_calc += id_torques.tail<2>();
#endif

  // The output for calculated torques.
  output_calc.head<2>() = torque_calc;
//  drake::log()->info("torque_calc: \n{}", torque_calc);
}

/// This is a helper system to distribute the output of the multifinger QP
/// controller to the separate finger force controllers.
class QPPlanToForceControllers : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QPPlanToForceControllers);
  QPPlanToForceControllers() {
    this->DeclareAbstractInputPort(
        "qp_control",
        Value<std::unordered_map<
            Finger, multibody::ExternallyAppliedSpatialForce<
                        double>>>{});  // std::map<Finger, ExtForces>
    this->DeclareAbstractOutputPort("f1_force", &QPPlanToForceControllers::CalcF1Output); // F1 std::vector<ExtForces>
    this->DeclareAbstractOutputPort("f2_force", &QPPlanToForceControllers::CalcF2Output); // F2 std::vector<ExtForces>
    this->DeclareAbstractOutputPort("f3_force", &QPPlanToForceControllers::CalcF3Output); // F3 std::vector<ExtForces>
  }

  void CalcF1Output(
      const systems::Context<double>& context,
      std::vector<multibody::ExternallyAppliedSpatialForce<double>>*
          desired_force_vec) const {
    desired_force_vec->clear();
    auto finger_force_map =
        this->get_input_port(0)
            .Eval<std::unordered_map<
                Finger, multibody::ExternallyAppliedSpatialForce<double>>>(
                context);
    multibody::ExternallyAppliedSpatialForce<double> desired_force;
    for (auto finger_force : finger_force_map) {
      if (finger_force.first == Finger::kFinger1) {
        desired_force = finger_force.second;
        break;
      }
    }
    desired_force_vec->push_back(desired_force);
  }

  void CalcF2Output(
      const systems::Context<double>& context,
      std::vector<multibody::ExternallyAppliedSpatialForce<double>>*
      desired_force_vec) const {
    desired_force_vec->clear();
    auto finger_force_map =
        this->get_input_port(0)
            .Eval<std::unordered_map<
                Finger, multibody::ExternallyAppliedSpatialForce<double>>>(
                context);
    multibody::ExternallyAppliedSpatialForce<double> desired_force;
    for (auto finger_force : finger_force_map) {
      if (finger_force.first == Finger::kFinger2) {
        desired_force = finger_force.second;
        break;
      }
    }
    desired_force_vec->push_back(desired_force);
  }

  void CalcF3Output(
      const systems::Context<double>& context,
      std::vector<multibody::ExternallyAppliedSpatialForce<double>>*
      desired_force_vec) const {
    desired_force_vec->clear();
    auto finger_force_map =
        this->get_input_port(0)
            .Eval<std::unordered_map<
                Finger, multibody::ExternallyAppliedSpatialForce<double>>>(
                context);
    multibody::ExternallyAppliedSpatialForce<double> desired_force;
    for (auto finger_force : finger_force_map) {
      if (finger_force.first == Finger::kFinger3) {
        desired_force = finger_force.second;
        break;
      }
    }
    desired_force_vec->push_back(desired_force);
  }
};

// TODO(rcory) Need to consolidate
// DoConnectFingerQPController/DoConnectGripperQPController.
void DoConnectGripperQPController(
    const MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph, lcm::DrakeLcm& lcm,
    const std::unordered_map<Finger, ForceController&>& finger_force_control_map,
    const ModelInstanceIndex& brick_index, const QPControlOptions qpoptions,
    std::map<std::string, const InputPort<double>&>& in_ports,
    std::map<std::string, const OutputPort<double>&>& out_ports,
    systems::DiagramBuilder<double>* builder) {
  // brick accel planned is 0. Use a constant source.
  auto brick_acceleration_planned_source =
      builder->AddSystem<systems::ConstantVectorSource<double>>(
          Eigen::Vector3d::Zero());

  // just use a constant target...
  Eigen::VectorXd des_state_vec(6);
  des_state_vec << 0, 0, qpoptions.thetaf_, 0, 0, 0;;
  auto state_traj_source =
      builder->AddSystem<systems::ConstantVectorSource<double>>(des_state_vec);

  // Spit out to scope
  systems::lcm::ConnectLcmScope(state_traj_source->get_output_port(),
                                "STATE_TRAJ", builder, &lcm);

  std::unordered_map<Finger, BrickFace>
      finger_face_assignment;
  finger_face_assignment.emplace(Finger::kFinger1, BrickFace::kNegY);
//  finger_face_assignment.emplace(Finger::kFinger2, BrickFace::kPosY);
//  finger_face_assignment.emplace(Finger::kFinger3, BrickFace::kNegZ);
  auto contact_face_source =
      builder->AddSystem<systems::ConstantValueSource<double>>(
          Value<std::unordered_map<Finger, BrickFace>>(finger_face_assignment));

  if (qpoptions.brick_only_) {
    // ================ Brick QP controller =================================
    // This implements the QP controller for brick only.
    // ======================================================================

    // Connect the QP controller
    builder->Connect(out_ports.at("plant_state"), in_ports.at("qp_state"));
    builder->Connect(out_ports.at("qp_control"),
                     in_ports.at("plant_spatial_force"));

    // TODO(rcory) Need to fix spatial force viz.
//    // To visualize the applied spatial forces.
//    auto viz_converter = builder->AddSystem<ExternalSpatialToSpatialViz>(
//        plant, brick_index, qpoptions.viz_force_scale_);
//    builder->Connect(out_ports.at("qp_control"), viz_converter->get_input_port(0));
//    builder->Connect(out_ports.at("brick_state"),
//                     viz_converter->get_input_port(1));
//    multibody::ConnectSpatialForcesToDrakeVisualizer(
//        builder, plant, viz_converter->get_output_port(0), &lcm);

    builder->Connect(contact_face_source->get_output_port(0),
                     in_ports.at("qp_finger_face_contact"));

    // Always make contact with sphere center at position (yc, zc).
    // TODO(rcory) bring back use of (yc, zc). Harcoded for now.

    // Specify the finger contact locations.
    auto f1_contact_pos = Eigen::Vector2d(-0.05, 0);
    auto f2_contact_pos = Eigen::Vector2d(0.05, 0);
    auto f3_contact_pos = Eigen::Vector2d(0, -0.05);
    auto f1_contact_src =
        builder->AddSystem<systems::ConstantVectorSource<double>>(f1_contact_pos);
    auto f2_contact_src =
        builder->AddSystem<systems::ConstantVectorSource<double>>(f2_contact_pos);
    auto f3_contact_src =
        builder->AddSystem<systems::ConstantVectorSource<double>>(f3_contact_pos);
    builder->Connect(f1_contact_src->get_output_port(),
                     in_ports.at("qp_f1_contact_pos"));
    builder->Connect(f2_contact_src->get_output_port(),
                     in_ports.at("qp_f2_contact_pos"));
    builder->Connect(f3_contact_src->get_output_port(),
                     in_ports.at("qp_f3_contact_pos"));

    builder->Connect(brick_acceleration_planned_source->get_output_port(),
                     in_ports.at("qp_desired_brick_accel"));
    builder->Connect(state_traj_source->get_output_port(),
                     in_ports.at("qp_desired_state"));
  } else {
    // ================ Planar Finger QP controller =========================
    // This implements the QP controller for brick AND planar-finger.
    // ======================================================================

    // Connect the QP controller
    builder->Connect(out_ports.at("plant_state"), in_ports.at("qp_state"));

    // TODO(rcory) Connect the output of the qp controller to the force
    //  controller.
    // Note: The spatial forces coming from the output of the QP controller
    // are already in the world frame (only the contact point is in the brick
    // frame)

    auto qp2force_sys = builder->AddSystem<QPPlanToForceControllers>();
    builder->Connect(out_ports.at("qp_control"),
                     qp2force_sys->GetInputPort("qp_control"));

    for (auto finger_force_control : finger_force_control_map) {
      if (finger_force_control.first == Finger::kFinger1) {
        builder->Connect(
            qp2force_sys->GetOutputPort("f1_force"),
            finger_force_control.second.get_force_desired_input_port());
      } else if (finger_force_control.first == Finger::kFinger2) {
        builder->Connect(
            qp2force_sys->GetOutputPort("f2_force"),
            finger_force_control.second.get_force_desired_input_port());
      } else if (finger_force_control.first == Finger::kFinger3) {
        builder->Connect(
            qp2force_sys->GetOutputPort("f3_force"),
            finger_force_control.second.get_force_desired_input_port());
      } else {
        throw std::runtime_error("Unknown Finger");
      }
    }

    // TODO(rcory) Need to fix this viz.
    unused(brick_index);
//    // To visualize the applied spatial forces.
//    auto viz_converter = builder->AddSystem<ExternalSpatialToSpatialViz>(
//        plant, brick_index, qpoptions.viz_force_scale_);
//    builder->Connect(out_ports.at("qp_control"),
//                     viz_converter->get_input_port(0));
//    multibody::ConnectSpatialForcesToDrakeVisualizer(
//        builder, plant, viz_converter->get_output_port(0), &lcm);
//    builder->Connect(out_ports.at("brick_state"),
//                     viz_converter->get_input_port(1));

    builder->Connect(contact_face_source->get_output_port(0),
                     in_ports.at("qp_contact_face"));

    // Adds a zero order hold to the contact results.
    auto zoh = builder->AddSystem<systems::ZeroOrderHold<double>>(
        1e-3, Value<ContactResults<double>>());
    builder->Connect(out_ports.at("contact_results"), zoh->get_input_port());

    // Adds system to calculate fingertip contact.
    for (auto finger_control : finger_force_control_map) {
        auto contact_point_calc_sys = builder->AddSystem<ContactPointInBrickFrame>(
            plant, scene_graph, finger_control.first);
        builder->Connect(zoh->get_output_port(),
                         contact_point_calc_sys->get_input_port(0));
        builder->Connect(out_ports.at("plant_state"),
                         contact_point_calc_sys->get_input_port(1));
        builder->Connect(contact_point_calc_sys->get_output_port(0),
                         in_ports.at("qp_p_BFingerTip"));
        builder->Connect(out_ports.at("scene_graph_query"),
                         contact_point_calc_sys->get_geometry_query_input_port());

      // Communicate the contact point to the force controller (if available).
      builder->Connect(contact_point_calc_sys->get_output_port(0),
                       finger_control.second.get_p_BrFingerTip_input_port());

      // Tells the force controller whether there is contact between the fingertip
      // and the brick.
      builder->Connect(contact_point_calc_sys->GetOutputPort("b_in_contact"),
                       finger_control.second.get_is_contact_input_port());

      // Provides contact point ref accelerations to the force controller (for
      // inverse dynamics).
      // TODO(rcory) likely don't need these accels anymore. Consider removing.
      auto v_BrCr = builder->AddSystem<systems::DiscreteDerivative>(2, 1e-3);
      auto a_BrCr = builder->AddSystem<systems::DiscreteDerivative>(2, 1e-3);
      builder->Connect(contact_point_calc_sys->get_output_port(0),
                       v_BrCr->get_input_port());
      builder->Connect(v_BrCr->get_output_port(), a_BrCr->get_input_port());
      builder->Connect(
          a_BrCr->get_output_port(),
          finger_control.second.get_contact_point_ref_accel_input_port());
    }

    builder->Connect(brick_acceleration_planned_source->get_output_port(),
                     in_ports.at("qp_desired_brick_accel"));
    builder->Connect(state_traj_source->get_output_port(),
                     in_ports.at("qp_desired_state"));
  }
}

void DoConnectFingerQPController(
    const MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph, lcm::DrakeLcm& lcm,
    const ForceController& force_controller,
    const ModelInstanceIndex& brick_index, const QPControlOptions qpoptions,
    std::map<std::string, const InputPort<double>&>& in_ports,
    std::map<std::string, const OutputPort<double>&>& out_ports,
    systems::DiagramBuilder<double>* builder) {
  // thetaddot_planned is 0. Use a constant source.
  auto thetaddot_planned_source =
      builder->AddSystem<systems::ConstantVectorSource<double>>(Vector1d(0));

  // The planned theta trajectory is from 0 to 90 degree in T seconds.
  std::vector<double> T = {-0.5, 0, qpoptions.T_, qpoptions.T_ + 0.5};
  std::vector<MatrixX<double>> Y(T.size(), MatrixX<double>::Zero(1, 1));
  Y[0](0, 0) = qpoptions.theta0_;
  Y[1](0, 0) = qpoptions.theta0_;
  Y[2](0, 0) = qpoptions.thetaf_;
  Y[3](0, 0) = qpoptions.thetaf_;

//  const trajectories::PiecewisePolynomial<double> theta_traj =
//      trajectories::PiecewisePolynomial<double>::Pchip(T, Y);

  // this doesn't work well for disturbed trajectories (since the traj is only
  // computed at initialization).
//  auto theta_traj_source =
//      builder->AddSystem<systems::TrajectorySource<double>>(
//          theta_traj, 1 /* take 1st derivatives */);

  // so, just use a constant target...
  auto theta_vec = Eigen::Vector2d(qpoptions.thetaf_, 0);
  auto theta_traj_source =
      builder->AddSystem<systems::ConstantVectorSource<double>>(theta_vec);

  // Spit out to scope
  systems::lcm::ConnectLcmScope(theta_traj_source->get_output_port(),
                                "THETA_TRAJ", builder, &lcm);

  // The face on the brick to make contact with.
  Value<BrickFace> contact_face;
  contact_face.set_value(qpoptions.contact_face_);
  auto contact_face_source =
      builder->AddSystem<systems::ConstantValueSource<double>>(contact_face);

  if (qpoptions.brick_only_) {
    // ================ Brick QP controller =================================
    // This implements the QP controller for brick only.
    // ======================================================================

    // Connect the QP controller
    builder->Connect(out_ports.at("plant_state"), in_ports.at("qp_state"));
    builder->Connect(out_ports.at("qp_control"),
                     in_ports.at("plant_spatial_force"));

    // To visualize the applied spatial forces.
    auto viz_converter = builder->AddSystem<ExternalSpatialToSpatialViz>(
        plant, brick_index, qpoptions.viz_force_scale_);
    builder->Connect(out_ports.at("qp_control"), viz_converter->get_input_port(0));
    builder->Connect(out_ports.at("brick_state"),
                     viz_converter->get_input_port(1));
    multibody::ConnectSpatialForcesToDrakeVisualizer(
        builder, plant, viz_converter->get_output_port(0), &lcm);

    builder->Connect(contact_face_source->get_output_port(0),
                     in_ports.at("qp_contact_face"));

    // Always make contact with sphere center at position (yc, zc).
    auto p_BCb_source =
        builder->AddSystem<systems::ConstantVectorSource<double>>(
            Eigen::Vector2d(qpoptions.yc_, qpoptions.zc_));
    builder->Connect(p_BCb_source->get_output_port(),
                     in_ports.at("qp_p_BFingerTip"));

    builder->Connect(thetaddot_planned_source->get_output_port(),
                     in_ports.at("qp_desired_thetaddot"));
    builder->Connect(theta_traj_source->get_output_port(),
                     in_ports.at("qp_desired_state"));
  } else {
    // ================ Planar Finger QP controller =========================
    // This implements the QP controller for brick AND planar-finger.
    // ======================================================================

    // Connect the QP controller
    builder->Connect(out_ports.at("plant_state"), in_ports.at("qp_state"));

    // TODO(rcory) Connect this to the force controller.
    // Note: The spatial forces coming from the output of the QP controller
    // are already in the world frame (only the contact point is in the brick
    // frame)

    builder->Connect(out_ports.at("qp_control"),
                     force_controller.get_force_desired_input_port());

    // To visualize the applied spatial forces.
    auto viz_converter = builder->AddSystem<ExternalSpatialToSpatialViz>(
        plant, brick_index, qpoptions.viz_force_scale_);
    builder->Connect(out_ports.at("qp_control"),
                     viz_converter->get_input_port(0));
    multibody::ConnectSpatialForcesToDrakeVisualizer(
        builder, plant, viz_converter->get_output_port(0), &lcm);
    builder->Connect(out_ports.at("brick_state"),
                     viz_converter->get_input_port(1));

    builder->Connect(contact_face_source->get_output_port(0),
                     in_ports.at("qp_contact_face"));

    // Adds a zero order hold to the contact results.
    auto zoh = builder->AddSystem<systems::ZeroOrderHold<double>>(
        1e-3, Value<ContactResults<double>>());
    builder->Connect(out_ports.at("contact_results"), zoh->get_input_port());

    // Adds system to calculate fingertip contact.
    auto contact_point_calc_sys = builder->AddSystem<ContactPointInBrickFrame>(
        plant, scene_graph, force_controller.get_options().finger_to_control_);
    builder->Connect(zoh->get_output_port(),
                     contact_point_calc_sys->get_input_port(0));
    builder->Connect(out_ports.at("plant_state"),
                     contact_point_calc_sys->get_input_port(1));
    builder->Connect(contact_point_calc_sys->get_output_port(0),
                     in_ports.at("qp_p_BFingerTip"));
    builder->Connect(out_ports.at("scene_graph_query"),
                     contact_point_calc_sys->get_geometry_query_input_port());

    // Communicate the contact point to the force controller.
    builder->Connect(contact_point_calc_sys->get_output_port(0),
                     force_controller.get_p_BrFingerTip_input_port());

    // Tells the force controller whether there is contact between the fingertip
    // and the brick.
    builder->Connect(contact_point_calc_sys->GetOutputPort("b_in_contact"),
                     force_controller.get_is_contact_input_port());

    // Provides contact point ref accelerations to the force controller (for
    // inverse dynamics).
    // TODO(rcory) likely don't need these accels anymore. Consider removing.
    auto v_BrCr = builder->AddSystem<systems::DiscreteDerivative>(2, 1e-3);
    auto a_BrCr = builder->AddSystem<systems::DiscreteDerivative>(2, 1e-3);
    builder->Connect(contact_point_calc_sys->get_output_port(0),
                     v_BrCr->get_input_port());
    builder->Connect(v_BrCr->get_output_port(), a_BrCr->get_input_port());
    builder->Connect(a_BrCr->get_output_port(),
                     force_controller.get_contact_point_ref_accel_input_port());

    builder->Connect(thetaddot_planned_source->get_output_port(),
                     in_ports.at("qp_desired_thetaddot"));
    builder->Connect(theta_traj_source->get_output_port(),
                     in_ports.at("qp_desired_state"));
  }
}

/// Adds the QP controller to the diagram and returns its corresponding
/// input/output ports.
// TODO(rcory) Can I consolidate AddGripperQPControllerToDiagram and
//  AddFingerQPControllerToDiagram?
void AddGripperQPControllerToDiagram(
    const MultibodyPlant<double>& plant,
    systems::DiagramBuilder<double>* builder,
    const QPControlOptions& qpoptions,
    std::map<std::string, const InputPort<double>&>* in_ports,
    std::map<std::string, const OutputPort<double>&>* out_ports) {

  // QP controller
  double Kp = qpoptions.QP_Kp_;
  double Kd = qpoptions.QP_Kd_;
  double weight_thetaddot_error = qpoptions.QP_weight_thetaddot_error_;
  double weight_f_Cb_B = qpoptions.QP_weight_f_Cb_B_;
  double mu = qpoptions.QP_mu_;
  double damping = qpoptions.brick_damping_;
  double I_B = qpoptions.brick_inertia_;

  BrickType brick_type = BrickType::PinBrick;  // hardcode pin brick for now
  Eigen::Vector2d zero2d = Eigen::Vector2d::Zero();
  InstantaneousContactForceQPController* qp_controller =
      builder->AddSystem<InstantaneousContactForceQPController>(
          brick_type, &plant, zero2d /* Kp_t */, zero2d /* Kd_t */, Kp, Kd,
          0 /* weight_a */, weight_thetaddot_error, weight_f_Cb_B, mu,
          0 /* trans_damping */, damping, I_B, 1 /* mass_B */);

  // Get the QP controller ports.
  // Output ports.
  out_ports->insert(std::pair<std::string, const OutputPort<double>&>(
      "qp_control", qp_controller->get_output_port_control()));

  // Input ports.
  in_ports->insert(std::pair<std::string, const InputPort<double>&>(
      "qp_state", qp_controller->get_input_port_estimated_state()));
  in_ports->insert(std::pair<std::string, const InputPort<double>&>(
      "qp_finger_face_contact",
      qp_controller->get_input_port_finger_contact()));
  in_ports->insert(std::pair<std::string, const InputPort<double>&>(
      "qp_desired_brick_accel",
      qp_controller->get_input_port_desired_brick_acceleration()));
  in_ports->insert(std::pair<std::string, const InputPort<double>&>(
      "qp_desired_state", qp_controller->get_input_port_desired_state()));

  in_ports->insert(std::pair<std::string, const InputPort<double>&>(
      "qp_f1_contact_pos", qp_controller->get_input_port_f1_contact_pos()));
  in_ports->insert(std::pair<std::string, const InputPort<double>&>(
      "qp_f2_contact_pos", qp_controller->get_input_port_f2_contact_pos()));
  in_ports->insert(std::pair<std::string, const InputPort<double>&>(
      "qp_f3_contact_pos", qp_controller->get_input_port_f3_contact_pos()));

}

void AddFingerQPControllerToDiagram(
    const MultibodyPlant<double>& plant,
    systems::DiagramBuilder<double>* builder,
    const QPControlOptions& qpoptions,
    std::map<std::string, const InputPort<double>&>* in_ports,
    std::map<std::string, const OutputPort<double>&>* out_ports) {

  // QP controller
  double Kp = qpoptions.QP_Kp_;
  double Kd = qpoptions.QP_Kd_;
  double weight_thetaddot_error = qpoptions.QP_weight_thetaddot_error_;
  double weight_f_Cb_B = qpoptions.QP_weight_f_Cb_B_;
  double mu = qpoptions.QP_mu_;
  double damping = qpoptions.brick_damping_;
  double I_B = qpoptions.brick_inertia_;

  PlanarFingerInstantaneousQPController* qp_controller =
      builder->AddSystem<PlanarFingerInstantaneousQPController>(
          &plant, Kp, Kd, weight_thetaddot_error, weight_f_Cb_B, mu, damping,
          I_B);

  // Get the QP controller ports.
  // Output ports.
  out_ports->insert(std::pair<std::string, const OutputPort<double>&>(
      "qp_control", qp_controller->get_output_port_control()));

  // Input ports.
  in_ports->insert(std::pair<std::string, const InputPort<double>&>(
      "qp_state", qp_controller->get_input_port_estimated_state()));
  in_ports->insert(std::pair<std::string, const InputPort<double>&>(
      "qp_contact_face", qp_controller->get_input_port_contact_face()));
  in_ports->insert(std::pair<std::string, const InputPort<double>&>(
      "qp_p_BFingerTip", qp_controller->get_input_port_p_BFingerTip()));
  in_ports->insert(std::pair<std::string, const InputPort<double>&>(
      "qp_desired_thetaddot",
      qp_controller->get_input_port_desired_thetaddot()));
  in_ports->insert(std::pair<std::string, const InputPort<double>&>(
      "qp_desired_state", qp_controller->get_input_port_desired_state()));
}

/// Creates the QP controller (type depending on whether we are simulating a
/// a standard plant or planar-gripper Diagram), and connects it to the force
/// controller.

// Connects a QP controller for a standard plant type simulation.
void ConnectQPController(const MultibodyPlant<double>& plant,
                         const geometry::SceneGraph<double>& scene_graph,
                         lcm::DrakeLcm& lcm,
                         const ForceController& force_controller,
                         const ModelInstanceIndex& brick_index,
                         const QPControlOptions qpoptions,
                         systems::DiagramBuilder<double>* builder) {
  // Create a std::map to hold all input/output ports.
  std::map<std::string, const OutputPort<double>&> out_ports;
  std::map<std::string, const InputPort<double>&> in_ports;

  // Output ports.
  out_ports.insert(std::pair<std::string, const OutputPort<double>&>(
      "brick_state", plant.get_state_output_port(brick_index)));
  out_ports.insert(std::pair<std::string, const OutputPort<double>&>(
      "plant_state", plant.get_state_output_port()));
  out_ports.insert(std::pair<std::string, const OutputPort<double>&>(
      "contact_results", plant.get_contact_results_output_port()));
  out_ports.insert(std::pair<std::string, const OutputPort<double>&>(
      "scene_graph_query", scene_graph.get_query_output_port()));

  // Input ports.
  in_ports.insert(std::pair<std::string, const InputPort<double>&>(
      "plant_spatial_force", plant.get_applied_spatial_force_input_port()));

  AddFingerQPControllerToDiagram(plant, builder, qpoptions, &in_ports,
                                 &out_ports);

  DoConnectFingerQPController(plant, scene_graph, lcm, force_controller,
                              brick_index, qpoptions, in_ports, out_ports,
                              builder);
}

// Connects a QP controller for a PlanarGripper diagram type simulation..
void ConnectQPController(
    PlanarGripper& planar_gripper, lcm::DrakeLcm& lcm,
    const std::unordered_map<Finger, ForceController&> finger_force_control_map,
    const QPType qp_type, const QPControlOptions qpoptions,
    systems::DiagramBuilder<double>* builder) {
  const MultibodyPlant<double>& plant = planar_gripper.get_multibody_plant();
  const ModelInstanceIndex brick_index = planar_gripper.get_brick_index();
  const SceneGraph<double>& scene_graph = planar_gripper.get_scene_graph();

  // Create a std::map to hold all input/output ports.
  std::map<std::string, const OutputPort<double>&> out_ports;
  std::map<std::string, const InputPort<double>&> in_ports;

  // Output ports.
  out_ports.insert(std::pair<std::string, const OutputPort<double>&>(
      "brick_state", planar_gripper.GetOutputPort("brick_state")));
  out_ports.insert(std::pair<std::string, const OutputPort<double>&>(
      "plant_state", planar_gripper.GetOutputPort("plant_state")));
  out_ports.insert(std::pair<std::string, const OutputPort<double>&>(
      "contact_results", planar_gripper.GetOutputPort("contact_results")));
  out_ports.insert(std::pair<std::string, const OutputPort<double>&>(
      "scene_graph_query", planar_gripper.GetOutputPort("scene_graph_query")));

  // Input ports.
  in_ports.insert(std::pair<std::string, const InputPort<double>&>(
      "plant_spatial_force", planar_gripper.GetInputPort("spatial_force")));

  if (qp_type == QPType::FingerQP) {
    AddFingerQPControllerToDiagram(plant, builder, qpoptions, &in_ports,
                                   &out_ports);
    DRAKE_DEMAND(finger_force_control_map.size() == 1);
    DoConnectFingerQPController(
        plant, scene_graph, lcm, finger_force_control_map.begin()->second,
        brick_index, qpoptions, in_ports, out_ports, builder);
  } else if (qp_type == QPType::GripperQP) {
    AddGripperQPControllerToDiagram(plant, builder, qpoptions, &in_ports,
                                    &out_ports);
    DoConnectGripperQPController(plant, scene_graph, lcm,
                                 finger_force_control_map, brick_index,
                                 qpoptions, in_ports, out_ports, builder);
  } else {
    throw std::logic_error("Unknown QPType");
  }

}

/**
 *  The methods below are temporarily in this file, but are meant to support
 *  QP/force control for the planar-gripper (3-finger) model, using a single
 *  finger.
 */

PlantStateToFingerStateSelector::PlantStateToFingerStateSelector(
    const MultibodyPlant<double>& plant, const Finger finger) {
  std::string fnum = to_string(finger);
  std::vector<multibody::JointIndex> joint_index_vector;
  joint_index_vector.push_back(
      plant.GetJointByName(fnum + "_BaseJoint").index());
  joint_index_vector.push_back(
      plant.GetJointByName(fnum + "_MidJoint").index());
  state_selector_matrix_ =
      plant.MakeStateSelectorMatrix(joint_index_vector);

  this->DeclareVectorInputPort(
      "plant_state",
      systems::BasicVector<double>(plant.num_positions() +
                                   plant.num_velocities()));
  this->DeclareVectorOutputPort(
      "finger_state", systems::BasicVector<double>(kNumJointsPerFinger * 2),
      &PlantStateToFingerStateSelector::CalcOutput);
}

void PlantStateToFingerStateSelector::CalcOutput(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {
  VectorX<double> plant_state =
      this->EvalVectorInput(context, 0)->get_value();
  output->get_mutable_value() = state_selector_matrix_ * plant_state;
}

FingersToPlantActuationMap::FingersToPlantActuationMap(
    const MultibodyPlant<double>& plant, const Finger finger) : finger_(finger) {
  std::vector<multibody::JointIndex> joint_index_vector;

  // Create the Sᵤ matrix.
  for (int i = 0; i < kNumFingers; i++) {
    joint_index_vector.push_back(
        plant.GetJointByName("finger" + std::to_string(i + 1) + "_BaseJoint")
            .index());
    joint_index_vector.push_back(
        plant.GetJointByName("finger" + std::to_string(i + 1) + "_MidJoint")
            .index());
  }
  actuation_selector_matrix_ =
      plant.MakeActuatorSelectorMatrix(joint_index_vector);
  actuation_selector_matrix_inv_ = actuation_selector_matrix_.inverse();

  this->DeclareVectorInputPort(
      "u_in",  /* in plant actuator ordering */
      systems::BasicVector<double>(kNumJoints));
;
  this->DeclareVectorInputPort(
      "u_fn", /* override value for finger_n actuation {fn_base_u, fn_mid_u} */
      systems::BasicVector<double>(kNumJointsPerFinger));

  this->DeclareVectorOutputPort("u_out",
                                systems::BasicVector<double>(kNumJoints),
                                &FingersToPlantActuationMap::CalcOutput);
}

void FingersToPlantActuationMap::CalcOutput(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {

  VectorX<double> u_all_in =
      this->EvalVectorInput(context, 0)->get_value();
  Vector2<double> u_fn_in =  /* {fn_base_u, fn_mid_u} */
      this->EvalVectorInput(context, 1)->get_value();

  // Reorder the gripper actuation to: {f1_base_u, f1_mid_u, ...}
  VectorX<double> u_s = actuation_selector_matrix_inv_ * u_all_in;

  // Replace finger n actuation values with the torque control actuation values.
  u_s.segment((to_num(finger_) - 1) * kNumJointsPerFinger, kNumJointsPerFinger) =
      u_fn_in;

  // Set the output.
  auto u = actuation_selector_matrix_ * u_s;
  output->get_mutable_value() = u;
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
