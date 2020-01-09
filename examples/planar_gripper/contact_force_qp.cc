#include "drake/examples/planar_gripper/contact_force_qp.h"

#include <limits>

#include "drake/solvers/solve.h"

namespace drake {
namespace examples {
namespace planar_gripper {
const double kInf = std::numeric_limits<double>::infinity();

Eigen::Matrix2d GetFrictionConeEdges(double mu, BrickFace brick_face) {
  Eigen::Matrix2d friction_cone_edges_B;
  switch (brick_face) {
    case BrickFace::kNegY: {
      friction_cone_edges_B << 1, 1, -mu, mu;
      break;
    }
    case BrickFace::kPosY: {
      friction_cone_edges_B << -1, -1, -mu, mu;
      break;
    }
    case BrickFace::kNegZ: {
      friction_cone_edges_B << -mu, mu, 1, 1;
      break;
    }
    case BrickFace::kPosZ: {
      friction_cone_edges_B << -mu, mu, -1, -1;
      break;
    }
    default: { throw std::runtime_error("Unknown face."); }
  }
  return friction_cone_edges_B;
}

InstantaneousContactForceQP::InstantaneousContactForceQP(
    const GripperBrickHelper<double>* gripper_brick,
    const Eigen::Ref<const Eigen::Vector2d>& p_WB_planned,
    const Eigen::Ref<const Eigen::Vector2d>& v_WB_planned,
    const Eigen::Ref<const Eigen::Vector2d>& a_WB_planned, double theta_planned,
    double thetadot_planned, double thetaddot_planned,
    const Eigen::Ref<const Eigen::Matrix2d>& Kp1,
    const Eigen::Ref<const Eigen::Matrix2d>& Kd1, double Kp2, double Kd2,
    const Eigen::Ref<const Vector6<double>>& brick_state,
    const std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>&
        finger_face_assignment,
    double weight_a, double weight_thetaddot, double weight_f_Cb)
    : gripper_brick_{gripper_brick}, prog_{new solvers::MathematicalProgram()} {
  std::map<Finger, Vector2<symbolic::Expression>> f_Cb_B;
  // Also compute the total contact torque.
  symbolic::Expression total_torque = 0;
  finger_face_contacts_.reserve(finger_face_assignment.size());
  for (const auto& finger_face : finger_face_assignment) {
    Vector2<symbolic::Variable> f_Cbi_B_edges =
        prog_->NewContinuousVariables<2>("f_Cb_B_edges");
    // The friction cone edge weights have to be non-negative.
    prog_->AddBoundingBoxConstraint(0, kInf, f_Cbi_B_edges);

    // Now compute the friction force from the friction cone edge weights
    const double mu =
        gripper_brick_->GetFingerTipBrickCoulombFriction(finger_face.first)
            .static_friction();
    // Now compute the edges of the friction cone in the brick frame. The edges
    // only depends on the brick face.
    const Eigen::Matrix2d friction_cone_edges_B =
        GetFrictionConeEdges(mu, finger_face.second.first);
    // Compute the finger contact position Cbi
    Vector2<double> p_BCbi = finger_face.second.second;
    Vector2<symbolic::Expression> f_Cbi_B =
        friction_cone_edges_B * f_Cbi_B_edges;
    f_Cb_B.emplace(finger_face.first, f_Cbi_B);
    total_torque += p_BCbi(0) * f_Cbi_B(1) - p_BCbi(1) * f_Cbi_B(0);
    finger_face_contacts_.emplace_back(
        finger_face.first, finger_face.second.first, f_Cbi_B_edges, mu, p_BCbi);
  }
  // First compute the desired acceleration.
  const Eigen::Vector2d p_WB = brick_state.head<2>();
  const Eigen::Vector2d v_WB = brick_state.segment<2>(3);
  const Eigen::Vector2d a_WB_des =
      Kp1 * (p_WB_planned - p_WB) + Kd1 * (v_WB_planned - v_WB) + a_WB_planned;
  const double theta = brick_state(2);
  const double thetadot = brick_state(5);
  const double thetaddot_des = Kp2 * (theta_planned - theta) +
                               Kd2 * (thetadot_planned - thetadot) +
                               thetaddot_planned;
  const double brick_mass =
      gripper_brick_->brick_frame().body().get_default_mass();

  const double cos_theta = std::cos(theta);
  const double sin_theta = std::sin(theta);
  Eigen::Matrix2d R_WB;
  R_WB << cos_theta, -sin_theta, sin_theta, cos_theta;
  // Total contact force in the brick frame.
  Vector2<symbolic::Expression> f_total_B(0, 0);
  for (const auto& face_contact_force : f_Cb_B) {
    f_total_B += face_contact_force.second;
  }
  Vector2<symbolic::Expression> a_WB =
      Eigen::Vector2d(
          0, 0*-multibody::UniformGravityFieldElement<double>::kDefaultStrength) +
      (R_WB * f_total_B) / brick_mass;
  const double I_B = dynamic_cast<const multibody::RigidBody<double>&>(
                         gripper_brick_->brick_frame().body())
                         .default_rotational_inertia()
                         .get_moments()(0);
  symbolic::Expression thetaddot = total_torque / I_B;
  prog_->AddQuadraticCost((a_WB - a_WB_des).squaredNorm() * weight_a);
  using std::pow;
  prog_->AddQuadraticCost(pow(thetaddot - thetaddot_des, 2) * weight_thetaddot);
  for (const auto& finger_contact_force : f_Cb_B) {
    prog_->AddQuadraticCost(finger_contact_force.second.squaredNorm() *
                            weight_f_Cb);
  }
}

std::unordered_map<Finger, std::pair<Eigen::Vector2d, Eigen::Vector2d>>
InstantaneousContactForceQP::GetFingerContactForceResult(
    const solvers::MathematicalProgramResult& result) const {
  std::unordered_map<Finger, std::pair<Eigen::Vector2d, Eigen::Vector2d>>
      finger_contact_forces;
  finger_contact_forces.reserve(finger_face_contacts_.size());
  for (const auto& finger_face_contact : finger_face_contacts_) {
    const Eigen::Vector2d f_Cbi_B_edges_sol =
        result.GetSolution(finger_face_contact.f_Cb_B_edges);
    const Eigen::Matrix2d friction_cone_edges_B = GetFrictionConeEdges(
        finger_face_contact.mu, finger_face_contact.brick_face);
    finger_contact_forces.emplace(
        finger_face_contact.finger,
        std::make_pair(
            Eigen::Vector2d(friction_cone_edges_B * f_Cbi_B_edges_sol),
            finger_face_contact.p_BCb_));
  }
  return finger_contact_forces;
}

InstantaneousContactForceQPController::InstantaneousContactForceQPController(
    const GripperBrickHelper<double>* gripper_brick,
    const Eigen::Ref<const Eigen::Matrix2d>& Kp1,
    const Eigen::Ref<const Eigen::Matrix2d>& Kd1, double Kp2, double Kd2,
    double weight_a, double weight_thetaddot, double weight_f_Cb_B)
    : gripper_brick_{gripper_brick},
      Kp1_{Kp1},
      Kd1_{Kd1},
      Kp2_{Kp2},
      Kd2_{Kd2},
      weight_a_{weight_a},
      weight_thetaddot_{weight_thetaddot},
      weight_f_Cb_B_{weight_f_Cb_B} {
  DRAKE_DEMAND(Kp2_ >= 0);
  DRAKE_DEMAND(Kd2_ >= 0);
  DRAKE_DEMAND(weight_a_ >= 0);
  DRAKE_DEMAND(weight_thetaddot_ >= 0);
  DRAKE_DEMAND(weight_f_Cb_B_ >= 0);

  output_index_control_ =
      this->DeclareAbstractOutputPort(
              "control", &InstantaneousContactForceQPController::CalcControl)
          .get_index();

  output_index_contact_force_ =
      this->DeclareAbstractOutputPort(
              "finger_contact_force",
              &InstantaneousContactForceQPController::CalcSpatialContactForce)
          .get_index();

  input_index_state_ =
      this->DeclareInputPort(systems::kVectorValued,
                             gripper_brick_->plant().num_positions() +
                                 gripper_brick_->plant().num_velocities())
          .get_index();

  input_index_desired_brick_state_ =
      this->DeclareInputPort(systems::kVectorValued, 6).get_index();
  input_index_desired_brick_acceleration_ =
      this->DeclareInputPort(systems::kVectorValued, 3).get_index();
  input_index_finger_contact_ =
      this->DeclareAbstractInputPort(
              "finger_contact",
              Value<std::unordered_map<
                  Finger, std::pair<BrickFace, Eigen::Vector2d>>>{})
          .get_index();
}

void InstantaneousContactForceQPController::CalcControl(
    const systems::Context<double>& context,
    std::unordered_map<Finger,
                       multibody::ExternallyAppliedSpatialForce<double>>*
        control) const {
  control->clear();
  const Eigen::VectorBlock<const VectorX<double>> state =
      get_input_port_estimated_state().Eval(context);
  const Eigen::VectorBlock<const VectorX<double>> desired_brick_state =
      get_input_port_desired_state().Eval(context);
  const Eigen::VectorBlock<const VectorX<double>> desired_brick_acceleration =
      get_input_port_desired_brick_acceleration().Eval(context);
  const std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>
      finger_contacts =
          get_input_port_finger_contact()
              .Eval<std::unordered_map<Finger,
                                       std::pair<BrickFace, Eigen::Vector2d>>>(
                  context);

  const Eigen::Vector2d p_WB_planned = desired_brick_state.head<2>();
  const Eigen::Vector2d v_WB_planned = desired_brick_state.segment<2>(3);
  const Eigen::Vector2d a_WB_planned = desired_brick_acceleration.head<2>();
  const double theta_planned = desired_brick_state(2);
  const double thetadot_planned = desired_brick_state(5);
  const double thetaddot_planned = desired_brick_acceleration(2);
  Vector6<double> brick_state;
  brick_state << state(gripper_brick_->brick_translate_y_position_index()),
      state(gripper_brick_->brick_translate_z_position_index()),
      state(gripper_brick_->brick_revolute_x_position_index()),
      state(gripper_brick_->plant().num_positions() +
            gripper_brick_->brick_translate_y_position_index()),
      state(gripper_brick_->plant().num_positions() +
            gripper_brick_->brick_translate_z_position_index()),
      state(gripper_brick_->plant().num_positions() +
            gripper_brick_->brick_revolute_x_position_index());

  InstantaneousContactForceQP qp(
      gripper_brick_, p_WB_planned, v_WB_planned, a_WB_planned, theta_planned,
      thetadot_planned, thetaddot_planned, Kp1_, Kd1_, Kp2_, Kd2_, brick_state,
      finger_contacts, weight_a_, weight_thetaddot_, weight_f_Cb_B_);

  const auto qp_result = solvers::Solve(qp.prog());
  const std::unordered_map<Finger, std::pair<Eigen::Vector2d, Eigen::Vector2d>>
      finger_contact_result = qp.GetFingerContactForceResult(qp_result);
  const double theta = brick_state(2);
  const double cos_theta = std::cos(theta);
  const double sin_theta = std::sin(theta);
  Eigen::Matrix2d R_WB;
  R_WB << cos_theta, -sin_theta, sin_theta, cos_theta;
  for (const auto& finger_contact : finger_contacts) {
    multibody::ExternallyAppliedSpatialForce<double> spatial_force;
    spatial_force.body_index = gripper_brick_->brick_frame().body().index();
    const std::pair<Eigen::Vector2d, Eigen::Vector2d> force_position =
        finger_contact_result.at(finger_contact.first);
    drake::log()->info("force: \n{}", force_position.first);
    drake::log()->info("position: \n{}", force_position.second);
    const Eigen::Vector2d p_BCb = force_position.second;
    spatial_force.p_BoBq_B = Eigen::Vector3d(0, p_BCb(0), p_BCb(1));
    const Eigen::Vector2d f_Cb_W = R_WB * force_position.first;
    spatial_force.F_Bq_W = multibody::SpatialForce<double>(
        Eigen::Vector3d::Zero(), Eigen::Vector3d(0, f_Cb_W(0), f_Cb_W(1)));
    control->emplace(finger_contact.first, spatial_force);
  }
}

void InstantaneousContactForceQPController::CalcSpatialContactForce(
    const systems::Context<double>& context,
    std::vector<multibody::ExternallyAppliedSpatialForce<double>>*
        contact_forces) const {
  std::unordered_map<Finger, multibody::ExternallyAppliedSpatialForce<double>>
      control;
  CalcControl(context, &control);
  contact_forces->clear();
  contact_forces->reserve(control.size());
  for (const auto& finger_contact_force : control) {
    contact_forces->push_back(finger_contact_force.second);
  }
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
