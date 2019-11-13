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
#include "drake/systems/primitives/discrete_derivative.h"


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
      finger_index_(plant.GetModelInstanceByName("planar_finger")),
      brick_index_(plant.GetModelInstanceByName("object")),
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
  contact_point_ref_accel_input_port_ = this->DeclareVectorInputPort(
      "contact point ref accel", systems::BasicVector<double>(2)).get_index();

  contact_results_input_port_ =
      this->DeclareAbstractInputPort("contact_results",
                                     Value<ContactResults<double>>{})
          .get_index();

  accelerations_actual_input_port_ =  // actual accelerations of the MBP (7 x 1)
      this->DeclareVectorInputPort("plant_vdot", systems::BasicVector<double>(
                                                     plant_.num_velocities()))
          .get_index();

  const int kDebuggingOutputs = 3;
  torque_output_port_ = this->DeclareVectorOutputPort(
                                "tau",
                                systems::BasicVector<double>(
                                    plant.num_actuators() + kDebuggingOutputs),
                                &ForceController::CalcTauOutput)
                            .get_index();

  // This provides the geometry query object, which can compute the witness
  // points between fingertip and box (used in impedance control).
  geometry_query_input_port_ = this->DeclareAbstractInputPort(
      "geometry_query", Value<geometry::QueryObject<double>>{}).get_index();
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

  auto brick_state = /* 2-element vector of the state of the brick */
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

  // Get the actual contact force.
  const auto& contact_results =
      get_contact_results_input_port().Eval<ContactResults<double>>(context);

  // The geometry query object to be used for impedance control (determines
  // closest points between fingertip and brick).
  // TODO(rcory) Consider moving this to ContactPointInBrickFrame system.
  geometry::QueryObject<double> geometry_query_obj =
      get_geometry_query_input_port().Eval<geometry::QueryObject<double>>(
          context);

  Eigen::Vector3d force_sim_W(0, 0, 0);
  // assume only zero or one contact is possible.
  if (contact_results.num_point_pair_contacts() > 0) {
    auto contact_info = contact_results.point_pair_contact_info(0);
    force_sim_W = contact_info.contact_force();
  }

  // Keep only the last two components of the force (since we only care about
  // forces in the y-z plane, i.e., in the plane of motion). This force returns
  // as the force felt by the brick.
//  Eigen::Vector2d force_act_yz_W = force_sim_W.tail<2>();

  // Set the plant's position and velocity within the context.
  plant_.SetPositionsAndVelocities(plant_context_.get(), finger_index_,
                                   finger_state);
  plant_.SetPositionsAndVelocities(plant_context_.get(), brick_index_,
                                   brick_state);

  // Initialize the vector for calculated torque commands.
  Eigen::Vector2d torque_calc(0, 0);

  // Gravity compensation.
  // TODO(rcory) why is this not the first two components of
  //  CalcGravityGeneralizedForces? Joint ordering indicates this should be
  //  the case...
   torque_calc =
       -plant_.CalcGravityGeneralizedForces(*plant_context_).segment<2>(1);
   multibody::MultibodyForces<double> external_forces(plant_);

//  multibody::MultibodyForces<double> external_forces(plant_);
//  plant_.CalcForceElementsContribution(
//      *plant_context_, &external_forces);
//  auto plant_vdot =
//      this->EvalVectorInput(context, accelerations_actual_input_port_)
//          ->get_value();
//  //drake::log()->info("vdot: \n{}", plant_vdot);
//  //Eigen::Vector3d plant_vdot(0, 0, 0);
//  auto vdot_ref = plant_vdot;
//  torque_calc = plant_.CalcInverseDynamics(
//      *plant_context_, vdot_ref, external_forces);

// Calculate the reference joint accelerations (based on contact reference accel.)


  // Compute the Jacobian.
  // For the finger/1-dof brick case, the plant consists of 3 dofs total (2 of
  // which belong to the finger). The resultant Jacobian will be of size 6 x 3.
  // This is the Jacobian of the finger tip contact reference (lies in the
  // intersection of geometry when there is contact), w.r.t. the finger base
  // frame. When there is no contact, it is the fingertip sphere center, w.r.t
  // the base frame.
  Eigen::Matrix<double, 6, 3> Jv_V_BaseFtip(6, 3);

  // p_WC is the contact point (reference) w.r.t. the World. When in contact
  // this reference is the actual contact point (lying inside the
  // contact intersection). When not in contact this reference is the
  // fingertip sphere center.
  Eigen::Vector3d p_LtFTip = Eigen::Vector3d::Zero();  /* ftip in tip link */
  Eigen::Vector3d p_WC = Eigen::Vector3d::Zero();
  const multibody::Frame<double>& tip_link_frame =
      plant_.GetBodyByName("finger_tip_link").body_frame();

  const multibody::Frame<double>& base_frame =
      plant_.GetBodyByName("finger_base").body_frame();

  const multibody::Frame<double>& brick_frame =
      plant_.GetBodyByName("brick_base_link").body_frame();

  // Desired forces (external spatial forces) are expressed in the world frame.
  // Express these in the brick frame instead (to compute the force error
  // terms), we then convert these commands to the finger base frame to match
  // the Jacobian.
  /* Rotation of world (W) w.r.t. finger brick (Br) */
  auto R_BrW = plant_.CalcRelativeRotationMatrix(
      *plant_context_, brick_frame, plant_.world_frame());
  Eigen::Vector3d force_des_Br = R_BrW * force_des_W;

  // Rotate the force reported by simulation to brick frame.
  Eigen::Vector3d force_act_Br = R_BrW * force_sim_W;

  // If we have contact, then the contact point reference is given by the
  // contact results object and it lies inside the intersection of the
  // fingertip sphere and brick geometries.
  if (contact_results.num_point_pair_contacts() > 0) {
    p_WC = contact_results.point_pair_contact_info(0).contact_point();
    plant_.CalcPointsPositions(*plant_context_, plant_.world_frame(), p_WC,
                               tip_link_frame, &p_LtFTip);
  } else {  // otherwise we have no contact, and we take the fingertip sphere
    // center as the contact point reference.
    p_LtFTip = GetFingerTipSpherePositionInLt(plant_, scene_graph_);
    plant_.CalcPointsPositions(*plant_context_, tip_link_frame, p_LtFTip,
                               plant_.world_frame(), &p_WC);
  }

  plant_.CalcJacobianSpatialVelocity(
      *plant_context_, multibody::JacobianWrtVariable::kV, tip_link_frame,
      p_LtFTip, base_frame, base_frame, &Jv_V_BaseFtip);

  // Extract the translational part of the Jacobian.
  // The last two three of Jv_V_WFtip correspond to x-y-z.
  // TODO(rcory) Figure out jacobian ordering.
  // Last two columns of Jacobian correspond to j1 and j2 (first column seems to
  // correspond to the brick).
  Eigen::Matrix<double, 3, 2> J_Ba(3, 2);
  J_Ba.block<1, 2>(0, 0) = Jv_V_BaseFtip.block<1, 2>(3, 1);
  J_Ba.block<1, 2>(1, 0) = Jv_V_BaseFtip.block<1, 2>(4, 1);
  J_Ba.block<1, 2>(2, 0) = Jv_V_BaseFtip.block<1, 2>(5, 1);

  // Force control gains. Allow force control in both (brick frame) y and z.
  Eigen::Matrix<double, 3, 3> Kf(3, 3);
  Kf << 0, 0, 0,  /* don't care about x */
        0, options_.kfy_, 0,
        0, 0, options_.kfz_;

  // Regulate force (in brick frame)
  Eigen::Vector3d delta_f_Br = force_des_Br - force_act_Br;
  // auto fy_command = Kf * delta_f + force_des;  //TODO(rcory) bug?
  Eigen::Vector3d force_error_command_Br = Kf * delta_f_Br;

  // auto fy_command = Kf Δf + Ki ∫ Δf dt - Kp p_e + f_d  // More general.
//  Eigen::Vector3d force_integral_command_Br = ;

  // Regulate position (in brick frame). First, rotate the contact point
  // reference into the brick frame.
  Eigen::Vector3d p_BrC = R_BrW * p_WC;

  // v_Ftip_Ba is the translational velocity of the fingertip contact reference
  // w.r.t. the finger base (Ba)
  Eigen::Vector3d v_Ftip_Ba = J_Ba * finger_state.segment<2>(2);
  // Convert fingertip translation velocity to brick frame.
  Eigen::Vector3d v_Ftip_Br = R_BrW * v_Ftip_Ba;
  // Compute the errors.
  Eigen::Vector3d delta_pos_Br =
      contact_ref_state_desired_Br.head<3>() - p_BrC.tail<3>();
  Eigen::Vector3d delta_vel_Br =
      contact_ref_state_desired_Br.tail<3>() - v_Ftip_Br.head<3>();
  Eigen::Matrix<double, 3, 3> Kp_pos(3, 3), Kd_pos(3, 3);
  Kp_pos.Zero();  // don't regulate position.
  Kd_pos << 0, 0, 0, 0, options_.kdy_, 0, 0, 0, options_.kdz_;  // regulate velocity in z (brick frame)
  Vector3d position_error_command_Br = Kp_pos * delta_pos_Br + Kd_pos * delta_vel_Br;

  // Extract the planar only (y-z) translational jacobian from the 3D (x,y,z)
  // translational jacobian.
  Eigen::Matrix<double, 2, 2> J_planar_Ba = J_Ba.block<2, 2>(1, 0);

  if (options_.always_direct_force_control_ ||
      contact_results.num_point_pair_contacts() > 0) {
    // Adds Joint damping.
    torque_calc += -options_.Kd_ * finger_state.segment<2>(2);

    // Torque due to hybrid position/force control
    /* Rotation of brick frame (Br) w.r.t. finger base frame (Ba) */
    auto R_BaBr = plant_.CalcRelativeRotationMatrix(
        *plant_context_, base_frame, brick_frame);
    Eigen::Vector3d force_command_Ba = (R_BaBr * force_des_Br) +
                                       (R_BaBr * force_error_command_Br) +
                                       (R_BaBr * position_error_command_Br);

    torque_calc += J_planar_Ba.transpose() * force_command_Ba.tail<2>();

    // TODO(rcory) only for debugging.
//    unused(force_command_Ba);
//    torque_calc += J_planar_Ba.transpose() * Eigen::Vector2d(0, -50);
  } else {
    // First, obtain the closest point on the brick from the fingertip sphere.
    // TODO(rcory) Consider moving this calculation to the system
    //  ContactPointInBrickFrame (in finger_brick.cc)
    auto pairs_vec = geometry_query_obj.ComputeSignedDistancePairwiseClosestPoints();
    DRAKE_DEMAND(pairs_vec.size() == 1);

    geometry::GeometryId brick_id = GetBrickGeometryId(plant_, scene_graph_);
    geometry::GeometryId ftip_id =
        GetFingerTipGeometryId(plant_, scene_graph_);

    Eigen::Vector2d target_position_Br;
    if (pairs_vec[0].id_A == ftip_id) {
      DRAKE_DEMAND(brick_id == pairs_vec[0].id_B);
      target_position_Br = pairs_vec[0].p_BCb.tail<2>();
    } else {
      DRAKE_DEMAND(brick_id == pairs_vec[0].id_A);
      target_position_Br = pairs_vec[0].p_ACa.tail<2>();
    }

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
    Vector2<double> imp_force_desired(y_force_desired_Br, z_force_desired_Br);
    torque_calc += J_planar_Ba.transpose() * imp_force_desired;
  }

  // The output for calculated torques.
  output_calc.head<2>() = torque_calc;

  // These are just auxiliary debugging outputs.
  output_calc.segment<2>(2) = Eigen::Vector2d::Zero();
  output_calc(4) = 0;
}

/// Creates the QP controller (type depending on whether we are simulating a
/// single brick or finger/brick), and connects it to the force controller.
void ConnectControllers(const MultibodyPlant<double>& plant,
                        const geometry::SceneGraph<double>& scene_graph,
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
    auto contact_point_calc_sys = builder->AddSystem<ContactPointInBrickFrame>(
        plant, scene_graph, options.yc_, options.zc_);
    builder->Connect(zoh->get_output_port(),
                     contact_point_calc_sys->get_input_port(0));
    builder->Connect(plant.get_state_output_port(),
                     contact_point_calc_sys->get_input_port(1));
    builder->Connect(contact_point_calc_sys->get_output_port(0),
                     qp_controller->get_input_port_p_BFingerTip());
    builder->Connect(scene_graph.get_query_output_port(),
                    contact_point_calc_sys->get_geometry_query_input_port());

    // Provides contact point ref accelerations to the force controller (for
    // inverse dynamics).
    auto v_BrCr = builder->AddSystem<systems::DiscreteDerivative>(2, 1e-3);
    auto a_BrCr = builder->AddSystem<systems::DiscreteDerivative>(2, 1e-3);
    builder->Connect(contact_point_calc_sys->get_output_port(0),
                     v_BrCr->get_input_port());
    builder->Connect(v_BrCr->get_output_port(), a_BrCr->get_input_port());
    builder->Connect(a_BrCr->get_output_port(),
                     force_controller.get_contact_point_ref_accel_input_port());

    builder->Connect(thetaddot_planned_source->get_output_port(),
                     qp_controller->get_input_port_desired_thetaddot());
    builder->Connect(theta_traj_source->get_output_port(),
                     qp_controller->get_input_port_desired_state());
  }
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
