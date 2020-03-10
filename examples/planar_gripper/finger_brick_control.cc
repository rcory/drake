#include "drake/examples/planar_gripper/finger_brick_control.h"
#include "drake/examples/planar_gripper/planar_gripper_lcm.h"
#include "drake/examples/planar_gripper/planar_gripper_udp.h"
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
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/examples/planar_gripper/contact_force_qp.h"
#include "drake/multibody/plant/spatial_forces_to_lcm.h"
#include "drake/systems/primitives/zero_order_hold.h"
#include "drake/systems/primitives/discrete_derivative.h"
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
using multibody::SpatialForce;

// Force controller with pure gravity compensation (no dynamics compensation
// yet).
// TODO(rcory) implement dynamic compensation.
ForceController::ForceController(const MultibodyPlant<double>& plant,
                                 const SceneGraph<double>& scene_graph,
                                 ForceControlOptions options)
    : plant_(plant),
      scene_graph_(scene_graph),
      options_(options) {
  // Make context with default parameters.
  plant_context_ = plant.CreateDefaultContext();

  // TODO(rcory) Can this just be a 2d force vector?
  force_desired_input_port_ =
      this->DeclareAbstractInputPort(
              "force_desired", Value<std::vector<
                         multibody::ExternallyAppliedSpatialForce<double>>>())
          .get_index();
  // Actual state of a single finger: {pos, vel}.
  finger_state_actual_input_port_ =
      this->DeclareVectorInputPort(
              "finger_state_actual",
              systems::BasicVector<double>(kNumJointsPerFinger * 2))
          .get_index();
  plant_state_actual_input_port_ =  // actual state of the entire plant
      this->DeclareVectorInputPort(
              "plant_state_actual",
              systems::BasicVector<double>(plant.num_multibody_states()))
          .get_index();
  // Contains the desired state of the fingertip (x, y, z, xdot, ydot, zdot)
  // measured in the world (W) frame.
  // Note: We ignore the x-components in the controller since the task is
  // planar.
  contact_state_desired_input_port_ =
      this->DeclareVectorInputPort(
              "tip_state_desired", systems::BasicVector<double>(6))
          .get_index();

  // Contact point reference acceleration (for ID control).
  // TODO(rcory) likely don't need this, consider removing this input port.
  contact_point_ref_accel_input_port_ = this->DeclareVectorInputPort(
      "contact_point_ref_accel", systems::BasicVector<double>(2)).get_index();

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

  torque_output_port_ =
      this->DeclareVectorOutputPort(
              "tau", systems::BasicVector<double>(kNumJointsPerFinger),
              &ForceController::CalcTauOutput)
          .get_index();

  p_BrCb_input_port_ =
      this->DeclareInputPort("p_BrCb", systems::kVectorValued, 2).get_index();

  is_contact_input_port_ =
      this->DeclareAbstractInputPort("is_in_contact",
                                     Value<bool>{}).get_index();

  this->DeclareContinuousState(3);  // stores ∫ Δf dt
}

void ForceController::CalcTauOutput(
    const systems::Context<double>& context,
    systems::BasicVector<double>* output_vector) const {
  auto output_calc = output_vector->get_mutable_value();
  output_calc.setZero();

//  drake::log()->info("Finger: {}", to_string(options_.finger_to_control_));

  // The finger to control.
  std::string finger_num = to_string(options_.finger_to_control_);

  // Run through the port evaluations.

  // This input contains the results of the QP planner, i.e., a desired force
  // applied at a contact point.
  auto external_spatial_forces_vec =
      get_force_desired_input_port()
          .Eval<std::vector<multibody::ExternallyAppliedSpatialForce<double>>>(
              context);
  DRAKE_DEMAND(external_spatial_forces_vec.size() == 1);
  // The desired fingertip force expressed in the world (W) frame.
  /* Note: Even though we store a Vector3, we ignore the x component, since we
   * only care about forces in the y-z plane. */
  Eigen::Vector3d force_des_W =
      external_spatial_forces_vec[0].F_Bq_W.translational();

  // This is the state of the finger to be controlled.
  VectorX<double> finger_state(kNumJointsPerFinger * 2);
  finger_state =
      this->EvalVectorInput(context, finger_state_actual_input_port_)
          ->get_value();

  // The state vector of the planar-gripper model (entire plant w/ brick).
  VectorX<double> plant_state =
      this->EvalVectorInput(context, plant_state_actual_input_port_)
          ->get_value();

  // 6-element vector of the desired contact state {position, velocity} in the
  // brick frame (x_des, y_des, z_des, xdot_des, ydot_des, zdot_des). Note that
  // the controller ignores the x-component since the task is in the y-z plane.
  Vector6<double> contact_state_desired_Br =
      this->EvalVectorInput(context, contact_state_desired_input_port_)
          ->get_value();

  // Get the contact point reference translational accelerations (yddot, zddot),
  // for inverse dynamics control.
//  auto a_BrCr = /* 2-element vector of the accels of the contact point ref. */
//      this->EvalVectorInput(context, contact_point_ref_accel_input_port_)
//          ->get_value();
//  unused(a_BrCr);

  // TODO(rcory) Remove this input port once contact point estimation exists.
  const auto& contact_results =
      get_contact_results_input_port().Eval<ContactResults<double>>(context);

  Eigen::Vector3d force_sensor_vec_W =
      this->EvalVectorInput(context, force_sensor_input_port_)->get_value();

  // Get the plant vdot for dynamics inverse dynamics.
  // TODO(rcory) I don't need this input anymore(?). Consider removing.
//  auto plant_vdot =
//      this->EvalVectorInput(context, accelerations_actual_input_port_)
//          ->get_value();
//  unused(plant_vdot);

  // Set the plant's position and velocity within the context.
  plant_.SetPositionsAndVelocities(plant_context_.get(), plant_state);

  // Define some important frames.
  const multibody::Frame<double>& tip_link_frame =
      plant_.GetBodyByName(finger_num + "_tip_link").body_frame();

  const multibody::Frame<double>& base_frame =
      plant_.GetBodyByName(finger_num + "_base").body_frame();

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
  unused(R_BrBa);

  // Initialize the vector for calculated torque commands.
  VectorX<double> torque_calc =
      VectorX<double>::Zero(kNumJointsPerFinger);

  // Gravity compensation.
  // TODO(rcory) Create a selector matrix from finger_num and assign directly
  //  to torque_calc via a matrix multiplication.
  int base_joint_index =
      plant_.GetJointByName(finger_num + "_BaseJoint").velocity_start();
  int mid_joint_index =
      plant_.GetJointByName(finger_num + "_MidJoint").velocity_start();
  VectorX<double> gravity_vec = VectorX<double>::Zero(plant_.num_velocities());
  gravity_vec =
       -plant_.CalcGravityGeneralizedForces(*plant_context_);
  torque_calc(0) = gravity_vec(base_joint_index);
  torque_calc(1) = gravity_vec(mid_joint_index);

  // p_WC is the contact point reference w.r.t. the World (W). When in contact
  // this reference is the actual contact point (lying inside the contact
  // intersection for the point contact model). When not in contact this
  // reference is the fingertip sphere center.
  Eigen::Vector3d p_LtC = Eigen::Vector3d::Zero();  /* contact point ref. in tip link frame */
  Eigen::Vector3d p_WC = Eigen::Vector3d::Zero();  /* contact point ref. in world frame */

  // Check whether there is contact between the fingertip and the brick.
  const bool& is_contact = get_is_contact_input_port().Eval<bool>(context);

  // If we have contact, then the contact point reference is given by the
  // contact results object and it lies inside the intersection of the
  // fingertip sphere and brick geometries (i.e., the actual contact point).
  if (is_contact) {
    p_WC = GetFingerContactPoint(contact_results, options_.finger_to_control_);
    plant_.CalcPointsPositions(*plant_context_, plant_.world_frame(), p_WC,
                               tip_link_frame, &p_LtC);
  } else {  // otherwise we have no contact, and we take the fingertip sphere
    // center as the contact point reference.
    p_LtC = GetFingerTipSpherePositionInLt(plant_, scene_graph_,
                                              options_.finger_to_control_);
    plant_.CalcPointsPositions(*plant_context_, tip_link_frame, p_LtC,
                               plant_.world_frame(), &p_WC);
  }

  // Regulate position (in brick frame).
  // First, rotate the contact point reference into the brick frame.
  Eigen::Vector3d p_BrC = R_BrW * p_WC;

  // Compute the translational velocity Jacobian.
  // For the finger/1-dof brick case, the plant consists of 3 dofs total (2 of
  // which belong to the finger). The resultant translational velocity Jacobian
  // will be of size 3 x 3. This is the Jacobian of the finger tip contact
  // reference (lies in the intersection of geometry when there is contact),
  // w.r.t. the finger base frame. When there is no contact, it is the fingertip
  // sphere center, w.r.t the base frame.
  MatrixX<double> Jv_V_BaC(3, plant_.num_velocities());
  plant_.CalcJacobianTranslationalVelocity(
      *plant_context_, multibody::JacobianWrtVariable::kV, tip_link_frame,
      p_LtC, base_frame, base_frame, &Jv_V_BaC);

  // Extract the 2 x 2 planar only (y-z) translational Jacobian that corresponds
  // to finger joints only (ignore the brick joint).
  Eigen::Matrix<double, 2, 2> J_planar_Ba(2, 2);
  J_planar_Ba.block<2, 1>(0, 0) =
      Jv_V_BaC.block<2, 1>(1, base_joint_index);
  J_planar_Ba.block<2, 1>(0, 1) = Jv_V_BaC.block<2, 1>(1, mid_joint_index);

  // Get the fingertip velocity in the brick frame.
  MatrixX<double> Jv_V_BrickFtip(3, plant_.num_velocities());
  plant_.CalcJacobianTranslationalVelocity(
      *plant_context_, multibody::JacobianWrtVariable::kV, tip_link_frame,
      p_LtC, brick_frame, brick_frame, &Jv_V_BrickFtip);
  Eigen::Vector3d v_BrC =
      Jv_V_BrickFtip * plant_.GetVelocities(*plant_context_);

//#define ADD_ACCELS
//#define SKIP_FORCE_CONT
#ifdef SKIP_FORCE_CONT
  unused(v_BrC);
  unused(p_BrC);
  unused(R_BaBr);
  unused(contact_state_desired_Br);
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
//    drake::log()->info("In direct force control.");
    // Desired forces (external spatial forces) are expressed in the world frame.
    // Express these in the brick frame instead (to compute the force error
    // terms), we then convert these commands to the finger base frame to match
    // the Jacobian.
    Eigen::Vector3d force_des_Br = R_BrW * force_des_W;

    // Get the control gains.
    Eigen::Matrix<double, 3, 3> Kp_force(3, 3);
    Eigen::Matrix<double, 3, 3> Ki_force(3, 3);
    Eigen::Matrix<double, 3, 3> Kp_position(3, 3);
    Eigen::Matrix<double, 3, 3> Kd_position(3, 3);
    GetGains(&Kp_force, &Ki_force, &Kp_position, &Kd_position);

    // Rotate the force reported by simulation to brick frame.
    Eigen::Vector3d force_act_Br = R_BrW * force_sensor_vec_W;

    // Regulate force (in brick frame)
    Eigen::Vector3d delta_f_Br = force_des_Br - force_act_Br;
    Eigen::Vector3d force_error_command_Br = Kp_force * delta_f_Br;

    // Integral error, which is stored in the continuous state.
    const systems::VectorBase<double>& state_vector =
        context.get_continuous_state_vector();
    const Eigen::VectorBlock<const VectorX<double>> state_block =
        dynamic_cast<const systems::BasicVector<double>&>(state_vector)
            .get_value();
    Eigen::Vector3d force_error_integral_Br = R_BrW * state_block;
    Eigen::Vector3d force_integral_command_Br =
        Ki_force * force_error_integral_Br;

    // Compute the errors.
    Eigen::Vector3d delta_pos_Br =
        contact_state_desired_Br.head<3>() - p_BrC;
    Eigen::Vector3d delta_vel_Br =
        contact_state_desired_Br.tail<3>() - v_BrC;
    Vector3d position_error_command_Br =
        Kp_position * delta_pos_Br + Kd_position * delta_vel_Br;

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
    torque_calc += -options_.Kd_joint_ * finger_state.segment<2>(2);

//    drake::log()->info("delta_pos_Br: \n{}", delta_pos_Br);
//    drake::log()->info("delta_vel_Br: \n{}", delta_vel_Br);
//    drake::log()->info("contact_state_des: \n{}", contact_state_desired_Br);
//    drake::log()->info("is_contact: \n{}", is_contact);
//    drake::log()->info("p_BrC: \n{}", p_BrC);
//    drake::log()->info("p_WC: \n{}", p_WC);
//    drake::log()->info("v_BrC: \n{}", v_BrC);

//    drake::log()->info("force_des_Br: \n{}", force_des_Br);
//    drake::log()->info("force_actual_Br: \n{}", force_act_Br);
//    drake::log()->info("force_delta_Br: \n{}", delta_f_Br);
//    drake::log()->info("force_error_command_Br: \n{}", force_error_command_Br);
//    drake::log()->info("force_error_integral_Br: \n{}", force_error_integral_Br);
//    drake::log()->info("position_error_command_Br: \n{}", position_error_command_Br);
//    drake::log()->info("force_integral_command_Br: \n{}", force_integral_command_Br);

    // Torque due to hybrid position/force control
    Eigen::Vector3d force_command_Ba = (R_BaBr * force_des_Br) +
                                       (R_BaBr * force_error_command_Br) +
                                       (R_BaBr * position_error_command_Br) +
                                       (R_BaBr * force_integral_command_Br);

    torque_calc += J_planar_Ba.transpose() * force_command_Ba.tail<2>();

    // TODO(rcory) only for debugging. Fixed desired force.
    // unused(force_command_Ba);
    // torque_calc += J_planar_Ba.transpose() * Eigen::Vector2d(0, -50);
//    drake::log()->info("torque_calc in DFC: {}", torque_calc.transpose());
  } else {  // Second Case: impedance control back to the brick's surface.
    // First, obtain the closest point on the brick from the fingertip sphere
    // center.
//    drake::log()->info("In impedance force control.");

    // Since we're not in contact, the p_BrCb input port holds the point on the
    // brick that is closest to the fingertip sphere center.
    Eigen::Vector2d p_BrC_des =
        get_p_BrCb_input_port().Eval(context);

//    Eigen::Vector3d p_BaC_des = Eigen::Vector3d::Zero();
//    p_BaC_des.tail<2>() = R_BaBr * p_BrC_des;
//    Eigen::Vector3d p_BaC = Eigen::Vector3d::Zero();
//    p_BaC = R_BaBr * p_BrC;

//    // Compute the desired contact point velocity.
//    // TODO(rcory) take in the brick's model instance.
//    ModelInstanceIndex brick_instance = plant_.GetModelInstanceByName("brick");
//    Vector1d brick_rotational_velocity =
//        plant_.GetVelocities(*plant_context_, brick_instance);
//    Eigen::Vector3d omega(brick_rotational_velocity(0), 0, 0);
//    unused(omega);
//    Eigen::Vector3d r = Eigen::Vector3d::Zero();
//    r.tail<2>() = p_BrC_des;
//    Eigen::Vector3d v_BrC_des = r.cross(omega);
//    drake::log()->info("v_BrC_des: \n{}", v_BrC_des);

//    drake::log()->info("finger_state: \n{}", finger_state);
//    drake::log()->info("brick_state: \n{}", brick_state);
//    drake::log()->info("J_planar_Ba: \n{}", J_planar_Ba);
//    drake::log()->info("p_BrFingerTip: \n{}", p_BrC_des);
//    drake::log()->info("p_BrC: \n{}", p_BrC);
//    drake::log()->info("v_Ftip_Ba: \n{}", v_Ftip_Ba);
//    drake::log()->info("v_BrC: \n{}", v_BrC);

    // Regulate the fingertip position back to the surface w/ impedance control
    // implement simple spring and damping laws for now (-kx - dxdot), i.e.,
    // just damped compliance control.
    const double K = options_.K_compliance_; // spring const
    const double D = options_.D_damping_; // damper
    double y_force_desired_Br =
        K * (p_BrC_des[0] - p_BrC(1)) - D * v_BrC(1);
    double z_force_desired_Br =
        K * (p_BrC_des[1] - p_BrC(2)) - D * v_BrC(2);
    Vector3<double> imp_force_desired_Br(0, y_force_desired_Br,
                                         z_force_desired_Br);

//    double y_force_desired_Ba =
//        K * (p_BaC_des[0] - p_BaC(1)) + D * (R_BaBr * v_BrC(1));
//    double z_force_desired_Ba =
//        K * (p_BaC_des[1] - p_BaC(2)) + D * (v_BrC_des(2) - v_BrC(2));
//    Vector3<double> imp_force_desired_Ba(0, y_force_desired_Ba,
//                                         z_force_desired_Ba);

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
  output_calc = torque_calc;
//  drake::log()->info("torque_calc: \n{}", torque_calc);
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
    controlled_force_diff = (force_des_W - force_sensor_vec_W);
//    drake::log()->info("force_des_W: \n{}", force_des_W);
//    drake::log()->info("force_sensor_vec_W: \n{}", force_sensor_vec_W);
//    drake::log()->info("controlled_force_diff: \n{}", controlled_force_diff);
  } else {
    // Intergral error, which is stored in the continuous state.
    // Drive the integral error to zero when there is no contact (i.e., reset).
    const systems::VectorBase<double>& state_vector =
        context.get_continuous_state_vector();
    const Eigen::VectorBlock<const VectorX<double>> state_block =
        dynamic_cast<const systems::BasicVector<double>&>(state_vector)
            .get_value();
    const double kIntegralLeakGain = 10.0;
    controlled_force_diff = -kIntegralLeakGain * state_block;
  }
  derivatives_vector.SetFromVector(controlled_force_diff);
}

void ForceController::GetGains(EigenPtr<Matrix3<double>> Kp_force,
                               EigenPtr<Matrix3<double>> Ki_force,
                               EigenPtr<Matrix3<double>> Kp_position,
                               EigenPtr<Matrix3<double>> Kd_position) const {
  DRAKE_DEMAND(Kp_force != EigenPtr<Vector3d>(nullptr));
  DRAKE_DEMAND(Ki_force != EigenPtr<Vector3d>(nullptr));
  DRAKE_DEMAND(Kp_position != EigenPtr<Vector3d>(nullptr));
  DRAKE_DEMAND(Kd_position != EigenPtr<Vector3d>(nullptr));

  // Don't regulate position.
  *Kp_position = MatrixX<double>::Zero(3, 3);

  if (options_.brick_face_ == BrickFace::kPosZ ||
      options_.brick_face_ == BrickFace::kNegZ) {
    *Kp_force << 0, 0, 0,  /* don't care about x */
        0, options_.kpf_t_, 0,
        0, 0, options_.kpf_n_;

    *Ki_force << 0, 0, 0,  /* don't care about x */
        0, options_.kif_t_, 0,
        0, 0, options_.kif_n_;

    *Kd_position << 0, 0, 0,
        0, options_.kd_t_, 0,
        0, 0, options_.kd_n_;  // regulate velocity in z (brick frame)
  } else {
    *Kp_force << 0, 0, 0,  /* don't care about x */
        0, options_.kpf_n_, 0,
        0, 0, options_.kpf_t_;

    *Ki_force << 0, 0, 0,  /* don't care about x */
        0, options_.kif_n_, 0,
        0, 0, options_.kif_t_;

    *Kd_position << 0, 0, 0,
        0, options_.kd_n_, 0,  // regulate normal velocity (brick frame)
        0, 0, options_.kd_t_;

  }
}

Vector3d ForceController::GetFingerContactPoint(
    const ContactResults<double>& contact_results, const Finger finger) const {
  std::string body_name = to_string(finger) + "_tip_link";
  multibody::BodyIndex fingertip_link_index =
      plant_.GetBodyByName(body_name).index();
  Vector3d contact_point;
  bool bfound = false;
  for (int i = 0; i < contact_results.num_point_pair_contacts(); i++) {
    auto& point_pair_info = contact_results.point_pair_contact_info(i);
    if (point_pair_info.bodyA_index() == fingertip_link_index ||
        point_pair_info.bodyB_index() == fingertip_link_index) {
      contact_point = point_pair_info.contact_point();
      bfound = true;
      break;
    }
  }
  if (!bfound) {
    throw std::runtime_error("Could not find " + body_name +
                             " in contact results.");
  }
  return contact_point;
}

/// This is a helper system to distribute the output of the multi-finger QP
/// controller to the separate finger force controllers.
class QPPlanToForceControllers : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QPPlanToForceControllers);
  QPPlanToForceControllers(multibody::BodyIndex brick_index)
      : brick_index_(brick_index) {
    this->DeclareAbstractInputPort(
        "qp_fingers_control",
        Value<std::unordered_map<
            Finger, multibody::ExternallyAppliedSpatialForce<
                        double>>>{});  // std::map<Finger, ExtForces>

    // The output ports below are all of type std::vector<ExtForces>
    this->DeclareAbstractOutputPort("f1_force",
                                    &QPPlanToForceControllers::CalcF1Output);
    this->DeclareAbstractOutputPort("f2_force",
                                    &QPPlanToForceControllers::CalcF2Output);
    this->DeclareAbstractOutputPort("f3_force",
                                    &QPPlanToForceControllers::CalcF3Output);
  }

  void CalcF1Output(
      const systems::Context<double>& context,
      std::vector<multibody::ExternallyAppliedSpatialForce<double>>*
          desired_force_vec) const {
    desired_force_vec->clear();
    auto finger_force_map =
        this->GetInputPort("qp_fingers_control")
            .Eval<std::unordered_map<
                Finger, multibody::ExternallyAppliedSpatialForce<double>>>(
                context);

    // By default, apply zero force to the brick.
    multibody::ExternallyAppliedSpatialForce<double> desired_force;
    desired_force.body_index = brick_index_;
    desired_force.F_Bq_W =
        multibody::SpatialForce<double>(Eigen::VectorXd::Zero(6));

    // If the finger-force controller exists, update the desired force.
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
        this->GetInputPort("qp_fingers_control")
            .Eval<std::unordered_map<
                Finger, multibody::ExternallyAppliedSpatialForce<double>>>(
                context);
    // By default, apply zero force to the brick.
    multibody::ExternallyAppliedSpatialForce<double> desired_force;
    desired_force.body_index = brick_index_;
    desired_force.F_Bq_W =
        multibody::SpatialForce<double>(Eigen::VectorXd::Zero(6));

    // If the finger-force controller exists, update the desired force.
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
        this->GetInputPort("qp_fingers_control")
            .Eval<std::unordered_map<
                Finger, multibody::ExternallyAppliedSpatialForce<double>>>(
                context);
    // By default, apply zero force to the brick.
    multibody::ExternallyAppliedSpatialForce<double> desired_force;
    desired_force.body_index = brick_index_;
    desired_force.F_Bq_W =
        multibody::SpatialForce<double>(Eigen::VectorXd::Zero(6));

    // If the finger-force controller exists, update the desired force.
    for (auto finger_force : finger_force_map) {
      if (finger_force.first == Finger::kFinger3) {
        desired_force = finger_force.second;
        break;
      }
    }
    desired_force_vec->push_back(desired_force);
  }

 private:
  multibody::BodyIndex brick_index_;
};

void DoConnectGripperQPController(
    const MultibodyPlant<double>& plant,
    const geometry::SceneGraph<double>& scene_graph, lcm::DrakeLcm& lcm,
    const std::optional<std::unordered_map<Finger, ForceController&>>&
        finger_force_control_map,
    const ModelInstanceIndex& brick_index, const QPControlOptions& qpoptions,
    std::map<std::string, const InputPort<double>&>& in_ports,
    std::map<std::string, const OutputPort<double>&>& out_ports,
    systems::DiagramBuilder<double>* builder) {

  systems::ConstantVectorSource<double>* brick_acceleration_planned_source;
  systems::ConstantVectorSource<double>* brick_state_traj_source;

  if (qpoptions.brick_type_ == BrickType::PlanarBrick) {
    // brick accel planned is 0 {yddot, zddot, thetaddot}. Use a constant
    // source.
    brick_acceleration_planned_source =
        builder->AddSystem<systems::ConstantVectorSource<double>>(
            Eigen::Vector3d::Zero());
    builder->Connect(brick_acceleration_planned_source->get_output_port(),
                     in_ports.at("qp_desired_brick_accel"));

    // Just use a constant target...{y, z, theta, ydot, zdot, thetadot}
    Eigen::VectorXd des_state_vec(6);
    des_state_vec << 0, 0, qpoptions.thetaf_, 0, 0, 0;;
    brick_state_traj_source =
        builder->AddSystem<systems::ConstantVectorSource<double>>(
            des_state_vec);
    builder->Connect(brick_state_traj_source->get_output_port(),
                     in_ports.at("qp_desired_brick_state"));
  } else {
    // brick accel planned is 0 {thetaddot}. Use a constant source.
    brick_acceleration_planned_source =
        builder->AddSystem<systems::ConstantVectorSource<double>>(
            Vector1d::Zero());
    builder->Connect(brick_acceleration_planned_source->get_output_port(),
                     in_ports.at("qp_desired_brick_accel"));

    // Just use a constant state target...{theta, thetadot}
    Eigen::Vector2d des_state_vec(qpoptions.thetaf_, 0);
    brick_state_traj_source =
        builder->AddSystem<systems::ConstantVectorSource<double>>(
            des_state_vec);
    builder->Connect(brick_state_traj_source->get_output_port(),
                     in_ports.at("qp_desired_brick_state"));
  }

  // Spit out to scope
  systems::lcm::ConnectLcmScope(brick_state_traj_source->get_output_port(),
                                "BRICK_STATE_TRAJ", builder, &lcm);

  // Connect the plant state to the QP controller
  builder->Connect(out_ports.at("plant_state"),
                   in_ports.at("qp_estimated_plant_state"));

  // To visualize the applied spatial forces.
  auto viz_converter = builder->AddSystem<ExternalSpatialToSpatialViz>(
      plant, brick_index, qpoptions.viz_force_scale_);
  builder->Connect(out_ports.at("qp_brick_control"),
                   viz_converter->get_input_port(0));
  multibody::ConnectSpatialForcesToDrakeVisualizer(
      builder, plant, viz_converter->get_output_port(0), &lcm);
  builder->Connect(out_ports.at("brick_state"),
                   viz_converter->get_input_port(1));

  if (qpoptions.brick_only_) {
    builder->Connect(out_ports.at("qp_brick_control"),
                     in_ports.at("plant_spatial_force"));

    auto finger_face_assigments_source = builder->AddSystem<
        systems::ConstantValueSource<double>>(
        Value<std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>>(
            qpoptions.finger_face_assignments_));

    builder->Connect(finger_face_assigments_source->get_output_port(0),
                     in_ports.at("qp_finger_face_assignments"));

  } else {
    // Note: The spatial forces coming from the output of the QP controller
    // are already in the world frame (only the contact point is in the brick
    // frame)
    auto qp2force_sys =
        builder->AddSystem<QPPlanToForceControllers>(GetBrickBodyIndex(plant));
    builder->Connect(out_ports.at("qp_fingers_control"),
                     qp2force_sys->GetInputPort("qp_fingers_control"));

    // Adds a zero order hold to the contact results.
    auto zoh_contact_results =
        builder->AddSystem<systems::ZeroOrderHold<double>>(
            1e-3, Value<ContactResults<double>>());
    builder->Connect(out_ports.at("contact_results"),
                     zoh_contact_results->get_input_port());

    DRAKE_DEMAND(finger_force_control_map.has_value());

    // Converts contact points (3 inputs) to an unordered_map of
    // finger_face_assignments (1 output).
    std::vector<Finger> fingers;
    for (auto iter : *finger_force_control_map) {
      fingers.push_back(iter.first);
    }
    auto contact_points_to_finger_face_assignments =
        builder->AddSystem<ContactPointsToFingerFaceAssignments>(fingers);
    builder->Connect(contact_points_to_finger_face_assignments->GetOutputPort(
                         "finger_face_assignments"),
                     in_ports.at("qp_finger_face_assignments"));

    // Adds system to calculate fingertip contact point.
    for (auto finger_force_control : *finger_force_control_map) {
      auto contact_point_calc_sys =
          builder->AddSystem<ContactPointInBrickFrame>(
              plant, scene_graph, finger_force_control.first);
      builder->Connect(zoh_contact_results->get_output_port(),
                       contact_point_calc_sys->GetInputPort("contact_results"));
      builder->Connect(out_ports.at("plant_state"),
                       contact_point_calc_sys->GetInputPort("x"));

      // Feed the contact point and contact face information to the
      // ContactPointsToFingerFaceAssignments system.
      if (finger_force_control.first == Finger::kFinger1) {
        builder->Connect(
            qp2force_sys->GetOutputPort("f1_force"),
            finger_force_control.second.get_force_desired_input_port());
        builder->Connect(
            contact_point_calc_sys->GetOutputPort("p_BrCb"),
            contact_points_to_finger_face_assignments->GetInputPort(
                "finger1_contact_point"));
        const BrickFace face_val =
            qpoptions.finger_face_assignments_.at(finger_force_control.first)
                .first;
        auto face_src =
            builder->AddSystem<systems::ConstantValueSource<double>>(
                Value<BrickFace>(face_val));
        builder->Connect(
            face_src->get_output_port(0),
            contact_points_to_finger_face_assignments->GetInputPort(
                "finger1_contact_face"));
        builder->Connect(
            contact_point_calc_sys->GetOutputPort("b_in_contact"),
            contact_points_to_finger_face_assignments->GetInputPort(
                "finger1_b_in_contact"));
      } else if (finger_force_control.first == Finger::kFinger2) {
        builder->Connect(
            qp2force_sys->GetOutputPort("f2_force"),
            finger_force_control.second.get_force_desired_input_port());
        builder->Connect(
            contact_point_calc_sys->GetOutputPort("p_BrCb"),
            contact_points_to_finger_face_assignments->GetInputPort(
                "finger2_contact_point"));
        const BrickFace face_val =
            qpoptions.finger_face_assignments_.at(finger_force_control.first)
                .first;
        auto face_src =
            builder->AddSystem<systems::ConstantValueSource<double>>(
                Value<BrickFace>(face_val));
        builder->Connect(
            face_src->get_output_port(0),
            contact_points_to_finger_face_assignments->GetInputPort(
                "finger2_contact_face"));
        builder->Connect(
            contact_point_calc_sys->GetOutputPort("b_in_contact"),
            contact_points_to_finger_face_assignments->GetInputPort(
                "finger2_b_in_contact"));
      } else if (finger_force_control.first == Finger::kFinger3) {
        builder->Connect(
            qp2force_sys->GetOutputPort("f3_force"),
            finger_force_control.second.get_force_desired_input_port());
        builder->Connect(
            contact_point_calc_sys->GetOutputPort("p_BrCb"),
            contact_points_to_finger_face_assignments->GetInputPort(
                "finger3_contact_point"));
        const BrickFace face_val =
            qpoptions.finger_face_assignments_.at(finger_force_control.first)
                .first;
        auto face_src =
            builder->AddSystem<systems::ConstantValueSource<double>>(
                Value<BrickFace>(face_val));
        builder->Connect(
            face_src->get_output_port(0),
            contact_points_to_finger_face_assignments->GetInputPort(
                "finger3_contact_face"));
        builder->Connect(
            contact_point_calc_sys->GetOutputPort("b_in_contact"),
            contact_points_to_finger_face_assignments->GetInputPort(
                "finger3_b_in_contact"));
      } else {
        throw std::runtime_error("Unknown Finger");
      }

      builder->Connect(out_ports.at("scene_graph_query"),
                       contact_point_calc_sys->get_geometry_query_input_port());

      // Communicate the contact point to the force controller (if available).
      builder->Connect(
          contact_point_calc_sys->GetOutputPort("p_BrCb"),
          finger_force_control.second.get_p_BrCb_input_port());

      // Tells the force controller whether there is contact between the
      // fingertip and the brick.
      builder->Connect(contact_point_calc_sys->GetOutputPort("b_in_contact"),
                       finger_force_control.second.get_is_contact_input_port());

      // Provides contact point ref accelerations to the force controller (for
      // inverse dynamics).
      // TODO(rcory) likely don't need these accels anymore. Consider
      // removing.
      auto v_BrCr = builder->AddSystem<systems::DiscreteDerivative>(2, 1e-3);
      auto a_BrCr = builder->AddSystem<systems::DiscreteDerivative>(2, 1e-3);
      builder->Connect(contact_point_calc_sys->GetOutputPort("p_BrCb"),
                       v_BrCr->get_input_port());
      builder->Connect(v_BrCr->get_output_port(), a_BrCr->get_input_port());
      builder->Connect(
          a_BrCr->get_output_port(),
          finger_force_control.second.get_contact_point_ref_accel_input_port());
    }
  }
}

/// Adds the QP controller to the diagram and returns its corresponding
/// input/output ports.
void AddGripperQPControllerToDiagram(
    const MultibodyPlant<double>& plant,
    systems::DiagramBuilder<double>* builder,
    const QPControlOptions& qpoptions,
    std::map<std::string, const InputPort<double>&>* in_ports,
    std::map<std::string, const OutputPort<double>&>* out_ports) {

  // QP controller
  double Kp_ro = qpoptions.QP_Kp_ro_;
  double Kd_ro = qpoptions.QP_Kd_ro_;
  double weight_thetaddot_error = qpoptions.QP_weight_thetaddot_error_;
  double weight_acceleration_error = qpoptions.QP_weight_acceleration_error_;
  double weight_f_Cb_B = qpoptions.QP_weight_f_Cb_B_;
  double mu = qpoptions.QP_mu_;
  double rotational_damping = qpoptions.brick_rotational_damping_;
  double translational_damping = qpoptions.brick_translational_damping_;
  double I_B = qpoptions.brick_inertia_;
  double mass_B = qpoptions.brick_mass_;

  // TODO(rcory) Remove this once planar brick is fully supported.
  if (qpoptions.brick_type_ == BrickType::PlanarBrick) {
    throw std::logic_error("Planar Brick not supported yet.");
  }

  Eigen::Matrix2d zero_2d_mat = Eigen::Matrix2d::Zero();
  InstantaneousContactForceQPController* qp_controller =
      builder->AddSystem<InstantaneousContactForceQPController>(
          qpoptions.brick_type_, &plant, zero_2d_mat /* Kp_tr */,
          zero_2d_mat /* Kd_tr */, Kp_ro, Kd_ro, weight_acceleration_error,
          weight_thetaddot_error, weight_f_Cb_B, mu, translational_damping,
          rotational_damping, I_B, mass_B);

  // Insert a zero order hold on the output of the controller, so that its
  // output (and hence it's computation of the QP) is only pulled at the
  // specified control dt.
  auto fingers_control_zoh = builder->AddSystem<systems::ZeroOrderHold<double>>(
      qpoptions.plan_dt,
      Value<std::unordered_map<
          Finger, multibody::ExternallyAppliedSpatialForce<double>>>());
  builder->Connect(qp_controller->get_output_port_fingers_control(),
                   fingers_control_zoh->get_input_port());

  auto brick_control_zoh = builder->AddSystem<systems::ZeroOrderHold<double>>(
      qpoptions.plan_dt,
      Value<std::vector<multibody::ExternallyAppliedSpatialForce<double>>>());
  builder->Connect(qp_controller->get_output_port_brick_control(),
                   brick_control_zoh->get_input_port());

  // Get the QP controller ports.
  // Output ports.
  out_ports->insert(std::pair<std::string, const OutputPort<double>&>(
      "qp_fingers_control", fingers_control_zoh->get_output_port()));
  out_ports->insert(std::pair<std::string, const OutputPort<double>&>(
      "qp_brick_control", brick_control_zoh->get_output_port()));

  // Input ports.
  in_ports->insert(std::pair<std::string, const InputPort<double>&>(
      "qp_estimated_plant_state",
      qp_controller->get_input_port_estimated_state()));
  in_ports->insert(std::pair<std::string, const InputPort<double>&>(
      "qp_finger_face_assignments",
      qp_controller->get_input_port_finger_face_assignments()));
  in_ports->insert(std::pair<std::string, const InputPort<double>&>(
      "qp_desired_brick_accel",
      qp_controller->get_input_port_desired_brick_acceleration()));
  in_ports->insert(std::pair<std::string, const InputPort<double>&>(
      "qp_desired_brick_state",
      qp_controller->get_input_port_desired_brick_state()));
}

// Connects a QP controller to a PlanarGripper diagram type simulation.
void ConnectQPController(
    const PlanarGripper& planar_gripper, lcm::DrakeLcm& lcm,
    const std::optional<std::unordered_map<Finger, ForceController&>>&
        finger_force_control_map,
    const QPControlOptions qpoptions,
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
    AddGripperQPControllerToDiagram(plant, builder, qpoptions, &in_ports,
                                    &out_ports);
    DoConnectGripperQPController(plant, scene_graph, lcm,
                                 finger_force_control_map, brick_index,
                                 qpoptions, in_ports, out_ports, builder);
}

void ConnectLCMQPController(
    const PlanarGripper& planar_gripper, lcm::DrakeLcm& lcm,
    const std::optional<std::unordered_map<Finger, ForceController&>>&
        finger_force_control_map,
    const QPControlOptions& qpoptions,
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

  // Adds the LCM QP Controller to the diagram.
  auto qp_controller = builder->AddSystem<PlanarGripperQPControllerLCM>(
      planar_gripper.get_multibody_plant().num_multibody_states(),
      planar_gripper.get_num_brick_states(),
      planar_gripper.get_num_brick_velocities(),
      GetBrickBodyIndex(planar_gripper.get_multibody_plant()), &lcm,
      kGripperLcmPeriod /* publish period */);

  // Get the QP controller ports.

  // Output ports.
  out_ports.insert(std::pair<std::string, const OutputPort<double>&>(
      "qp_fingers_control",
      qp_controller->GetOutputPort("qp_fingers_control")));
  out_ports.insert(std::pair<std::string, const OutputPort<double>&>(
      "qp_brick_control", qp_controller->GetOutputPort("qp_brick_control")));

  // Input ports.
  in_ports.insert(std::pair<std::string, const InputPort<double>&>(
      "qp_estimated_plant_state",
      qp_controller->GetInputPort("qp_estimated_plant_state")));
  in_ports.insert(std::pair<std::string, const InputPort<double>&>(
      "qp_finger_face_assignments",
      qp_controller->GetInputPort("qp_finger_face_assignments")));
  in_ports.insert(std::pair<std::string, const InputPort<double>&>(
      "qp_desired_brick_accel",
      qp_controller->GetInputPort("qp_desired_brick_accel")));
  in_ports.insert(std::pair<std::string, const InputPort<double>&>(
      "qp_desired_brick_state",
      qp_controller->GetInputPort("qp_desired_brick_state")));

  // Connects the LCM QP controller.
  DoConnectGripperQPController(plant, scene_graph, lcm,
                               finger_force_control_map, brick_index, qpoptions,
                               in_ports, out_ports, builder);
}

void ConnectUDPQPController(
    const PlanarGripper& planar_gripper, lcm::DrakeLcm& lcm,
    const std::optional<std::unordered_map<Finger, ForceController&>>&
        finger_force_control_map,
    const QPControlOptions& qpoptions, int publisher_local_port,
    int publisher_remote_port, unsigned long publisher_remote_address,
    int receiver_local_port, systems::DiagramBuilder<double>* builder) {
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

  // Adds the UDP QP Controller to the diagram.

  auto qp_controller = builder->AddSystem<PlanarGripperQPControllerUDP>(
      planar_gripper.get_multibody_plant().num_multibody_states(),
      GetBrickBodyIndex(planar_gripper.get_multibody_plant()),
      planar_gripper.num_gripper_joints() / 2,
      planar_gripper.get_num_brick_states(),
      planar_gripper.get_num_brick_velocities(), publisher_local_port,
      publisher_remote_port, publisher_remote_address, receiver_local_port,
      kGripperUdpStatusPeriod);

  // Get the QP controller ports.

  // Output ports.
  out_ports.insert(std::pair<std::string, const OutputPort<double>&>(
      "qp_fingers_control",
      qp_controller->GetOutputPort("qp_fingers_control")));
  out_ports.insert(std::pair<std::string, const OutputPort<double>&>(
      "qp_brick_control", qp_controller->GetOutputPort("qp_brick_control")));

  // Input ports.
  in_ports.insert(std::pair<std::string, const InputPort<double>&>(
      "qp_estimated_plant_state",
      qp_controller->GetInputPort("qp_estimated_plant_state")));
  in_ports.insert(std::pair<std::string, const InputPort<double>&>(
      "qp_finger_face_assignments",
      qp_controller->GetInputPort("qp_finger_face_assignments")));
  in_ports.insert(std::pair<std::string, const InputPort<double>&>(
      "qp_desired_brick_accel",
      qp_controller->GetInputPort("qp_desired_brick_accel")));
  in_ports.insert(std::pair<std::string, const InputPort<double>&>(
      "qp_desired_brick_state",
      qp_controller->GetInputPort("qp_desired_brick_state")));

  // Connects the UDP QP controller.
  DoConnectGripperQPController(plant, scene_graph, lcm,
                               finger_force_control_map, brick_index, qpoptions,
                               in_ports, out_ports, builder);
}

ForceController* SetupForceController(
    const PlanarGripper& planar_gripper, DrakeLcm& lcm,
    const ForceControlOptions& foptions,
    systems::DiagramBuilder<double>* builder) {
  auto& plant = planar_gripper.get_multibody_plant();
  auto& scene_graph = planar_gripper.get_scene_graph();

  auto zoh_contact_results = builder->AddSystem<systems::ZeroOrderHold<double>>(
      1e-3, Value<ContactResults<double>>());

  std::vector<SpatialForce<double>> init_spatial_vec{
      SpatialForce<double>(Vector3<double>::Zero(), Vector3<double>::Zero())};
  auto zoh_reaction_forces = builder->AddSystem<systems::ZeroOrderHold<double>>(
      1e-3, Value<std::vector<SpatialForce<double>>>(init_spatial_vec));

  // TODO(rcory) remove this?
//  auto zoh_joint_accels = builder->AddSystem<systems::ZeroOrderHold<double>>(
//      1e-3, plant.num_velocities());

  Finger kFingerToControl = foptions.finger_to_control_;

  // Connect the force controller
  auto force_controller = builder->AddSystem<ForceController>(
      plant, scene_graph, foptions);
  auto plant_to_finger_state_sel =
      builder->AddSystem<PlantStateToFingerStateSelector>(
          planar_gripper.get_multibody_plant(), kFingerToControl);
  builder->Connect(planar_gripper.GetOutputPort("plant_state"),
                   plant_to_finger_state_sel->GetInputPort("plant_state"));
  builder->Connect(plant_to_finger_state_sel->GetOutputPort("finger_state"),
                   force_controller->get_finger_state_actual_input_port());
  builder->Connect(planar_gripper.GetOutputPort("plant_state"),
                   force_controller->GetInputPort("plant_state_actual"));
  // TODO(rcory) Make sure we are passing in the proper gripper state, once
  //  horizontal with gravity on is supported (due to additional "x" prismatic
  //  joint).
  builder->Connect(planar_gripper.GetOutputPort("contact_results"),
                   zoh_contact_results->get_input_port());
  builder->Connect(zoh_contact_results->get_output_port(),
                   force_controller->get_contact_results_input_port());
  // TODO(rcory) Remove these joint accelerations once I confirm they are not
  // needed.
//  builder->Connect(plant.get_joint_accelerations_output_port(),
//                  zoh_joint_accels->get_input_port());
//  builder->Connect(zoh_joint_accels->get_output_port(),
//                  force_controller->get_accelerations_actual_input_port());
  auto zero_accels_src =
      builder->AddSystem<systems::ConstantVectorSource<double>>(
          VectorX<double>::Zero(plant.num_velocities()));
  builder->Connect(zero_accels_src->get_output_port(),
                   force_controller->get_accelerations_actual_input_port());

  builder->Connect(planar_gripper.GetOutputPort("reaction_forces"),
                   zoh_reaction_forces->get_input_port());

  auto force_demux_sys =
      builder->AddSystem<ForceDemuxer>(plant, kFingerToControl);
  builder->Connect(zoh_contact_results->get_output_port(),
                   force_demux_sys->get_contact_results_input_port());
  builder->Connect(zoh_reaction_forces->get_output_port(),
                   force_demux_sys->get_reaction_forces_input_port());
  builder->Connect(planar_gripper.GetOutputPort("plant_state"),
                   force_demux_sys->get_state_input_port());

  // Provide the actual force sensor input to the force controller. This
  // contains the reaction forces (total wrench) at the sensor weld joint.
  builder->Connect(force_demux_sys->get_reaction_vec_output_port(),
                   force_controller->get_force_sensor_input_port());

  // Connect to the scope.
  systems::lcm::ConnectLcmScope(
      force_controller->get_torque_output_port(),
      "TORQUE_OUTPUT_F" + std::to_string(to_num(kFingerToControl)), builder, &lcm);

  // We don't regulate position for now (set these to zero).
  // 6-vector represents pos-vel for fingertip contact point x-y-z. The control
  // ignores the x-components.
  const Vector6<double> tip_state_des_vec = Vector6<double>::Zero();
  auto const_pos_src =
      builder->AddSystem<systems::ConstantVectorSource>(tip_state_des_vec);
  builder->Connect(const_pos_src->get_output_port(),
                   force_controller->get_contact_state_desired_input_port());

  return force_controller;
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

FingerToPlantActuationMap::FingerToPlantActuationMap(
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
      systems::BasicVector<double>(kNumGripperJoints));

  this->DeclareVectorInputPort(
      "u_fn", /* override value for finger_n actuation {fn_base_u, fn_mid_u} */
      systems::BasicVector<double>(kNumJointsPerFinger));

  this->DeclareVectorOutputPort("u_out",
                                systems::BasicVector<double>(kNumGripperJoints),
                                &FingerToPlantActuationMap::CalcOutput);
}

void FingerToPlantActuationMap::CalcOutput(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {

  VectorX<double> u_all_in =
      this->EvalVectorInput(context, 0)->get_value();
  Vector2<double> u_fn_in =  /* {fn_base_u, fn_mid_u} */
      this->EvalVectorInput(context, 1)->get_value();

  // Reorder the gripper actuation to: {f1_base_u, f1_mid_u, ...}
  VectorX<double> u_s = actuation_selector_matrix_inv_ * u_all_in;

  // Replace finger n actuation values with the torque control actuation values.
  u_s.segment((to_num(finger_) - 1) * kNumJointsPerFinger,
              kNumJointsPerFinger) = u_fn_in;

  // Set the output.
  auto u = actuation_selector_matrix_ * u_s;
  output->get_mutable_value() = u;
}

ForceControllersToPlantActuationMap::ForceControllersToPlantActuationMap(
    const MultibodyPlant<double>& plant,
    std::unordered_map<Finger, ForceController&> finger_force_control_map)
    : finger_force_control_map_(finger_force_control_map) {
  std::vector<multibody::JointIndex> joint_index_vector;

  /// Build an actuation selector matrix Sᵤ such that `u = Sᵤ⋅uₛ`, where u is
  /// the vector of actuation values for the full model (ordered by
  /// JointActuatorIndex) and uₛ is a vector of actuation values for the
  /// actuators acting on the joints listed by `joint_index_vector`
  for (auto & iter : finger_force_control_map) {
    Finger finger_m = iter.first;
    joint_index_vector.push_back(
        plant.GetJointByName(to_string(finger_m) + "_BaseJoint")
            .index());
    joint_index_vector.push_back(
        plant.GetJointByName(to_string(finger_m) + "_MidJoint")
            .index());

    /* Input port n contains finger_m actuation {fm_base_u, fm_mid_u} */
    InputPortIndex n = this->DeclareVectorInputPort(
        to_string(finger_m) + "_u_in",
        systems::BasicVector<double>(kNumJointsPerFinger)).get_index();
    input_port_index_to_finger_map_[n] = finger_m;
  }
  actuation_selector_matrix_ =  /* Sᵤ */
      plant.MakeActuatorSelectorMatrix(joint_index_vector);

  // The single output containing MBP actuation values (in the plant's
  // joint-actuator ordering).
  this->DeclareVectorOutputPort(
      "u_out", systems::BasicVector<double>(kNumGripperJoints),
      &ForceControllersToPlantActuationMap::CalcOutput);
}

void ForceControllersToPlantActuationMap::CalcOutput(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {
  const int u_in_size =
      static_cast<int>(input_port_index_to_finger_map_.size()) *
      kNumJointsPerFinger;
  VectorX<double> u_in(u_in_size);

  // Gather all the inputs
  int u_in_index = 0;
  for (auto & iter : input_port_index_to_finger_map_) {
    /* insert {fm_base_u, fm_mid_u} */
    u_in.segment<kNumJointsPerFinger>(u_in_index) =
        this->EvalVectorInput(context, iter.first)->get_value();
    u_in_index += 2;
  }

  // Set the output. Actuators not called out in the creation of Sᵤ are set to
  // zero.
  auto u_out = actuation_selector_matrix_ * u_in;
  output->get_mutable_value() = u_out;
}

void ForceControllersToPlantActuationMap::ConnectForceControllersToPlant(
    const PlanarGripper& planar_gripper,
    systems::DiagramBuilder<double>* builder) const {

  // Connect the force controllers
  for (auto & iter : input_port_index_to_finger_map_) {
    InputPortIndex input_port_index = iter.first;
    Finger finger = iter.second;
    ForceController& force_controller = finger_force_control_map_.at(finger);
    builder->Connect(force_controller.get_torque_output_port(),
                     this->get_input_port(input_port_index));
  }
  builder->Connect(this->GetOutputPort("u_out"),
                   planar_gripper.GetInputPort("actuation"));
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
