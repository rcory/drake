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
    const BrickType brick_type,
    const Eigen::Ref<const Eigen::Vector2d>& p_WB_planned,
    const Eigen::Ref<const Eigen::Vector2d>& v_WB_planned,
    const Eigen::Ref<const Eigen::Vector2d>& a_WB_planned, double theta_planned,
    double thetadot_planned, double thetaddot_planned,
    const Eigen::Ref<const Eigen::Matrix2d>& Kp_t,
    const Eigen::Ref<const Eigen::Matrix2d>& Kd_t, double Kp_r, double Kd_r,
    const Eigen::Ref<const Vector6<double>>& brick_state,
    const std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>&
        finger_face_assignments,
    double weight_a_error, double weight_thetaddot_error, double weight_f_Cb,
    double mu, double I_B, double brick_mass, double rotational_damping,
    double translational_damping)
    : prog_{new solvers::MathematicalProgram()} {
  // Extract the brick state.
  const Eigen::Vector2d p_WB = brick_state.head<2>();
  const Eigen::Vector2d v_WB = brick_state.segment<2>(3);
  const double theta = brick_state(2);
  const double thetadot = brick_state(5);

  std::map<Finger, Vector2<symbolic::Expression>> f_Cb_B;
  // Also compute the total contact torque.
  symbolic::Expression total_torque = 0;
  finger_face_contacts_.reserve(finger_face_assignments.size());
  for (const auto& finger_face_assignment : finger_face_assignments) {
    Vector2<symbolic::Variable> f_Cbi_B_edges =
        prog_->NewContinuousVariables<2>("f_Cb_B_edges");
    // The friction cone edge weights have to be non-negative.
    prog_->AddBoundingBoxConstraint(0, kInf, f_Cbi_B_edges);

    // Now compute the edges of the friction cone in the brick frame. The edges
    // only depends on the brick face.
    const Eigen::Matrix2d friction_cone_edges_B =
        GetFrictionConeEdges(mu, finger_face_assignment.second.first);
    // Compute the finger contact position Cbi
    Vector2<double> p_BCbi = finger_face_assignment.second.second;
    Vector2<symbolic::Expression> f_Cbi_B =
        friction_cone_edges_B * f_Cbi_B_edges;
    f_Cb_B.emplace(finger_face_assignment.first, f_Cbi_B);
    total_torque += p_BCbi(0) * f_Cbi_B(1) - p_BCbi(1) * f_Cbi_B(0);
    finger_face_contacts_.emplace_back(finger_face_assignment.first,
                                       finger_face_assignment.second.first,
                                       f_Cbi_B_edges, mu, p_BCbi);
  }

  // Add the damping term
  total_torque += -rotational_damping * thetadot;

  // First compute the desired acceleration.
  const Eigen::Vector2d a_WB_des = Kp_t * (p_WB_planned - p_WB) +
                                   Kd_t * (v_WB_planned - v_WB) + a_WB_planned;
  const double thetaddot_des = Kp_r * (theta_planned - theta) +
                               Kd_r * (thetadot_planned - thetadot) +
                               thetaddot_planned;

  const double cos_theta = std::cos(theta);
  const double sin_theta = std::sin(theta);
  Eigen::Matrix2d R_WB;
  R_WB << cos_theta, -sin_theta, sin_theta, cos_theta;
  // Total contact force in the brick frame.
  Vector2<symbolic::Expression> f_total_B(0, 0);
  for (const auto& face_contact_force : f_Cb_B) {
    f_total_B += face_contact_force.second;
  }

  Eigen::Matrix2d D;
  D << translational_damping, 0, 0, translational_damping;
  Vector2<symbolic::Expression> a_WB =
      Eigen::Vector2d(0, 0 * -multibody::UniformGravityFieldElement<
                                 double>::kDefaultStrength) +
      (R_WB * f_total_B - D * brick_state.segment<2>(3)) / brick_mass;

  symbolic::Expression thetaddot = total_torque / I_B;
  if (brick_type == BrickType::PlanarBrick) {
    prog_->AddQuadraticCost((a_WB - a_WB_des).squaredNorm() * weight_a_error);
  }
  using std::pow;
  prog_->AddQuadraticCost(pow(thetaddot - thetaddot_des, 2) *
                          weight_thetaddot_error);
  for (const auto& finger_contact_force : f_Cb_B) {
    prog_->AddQuadraticCost(finger_contact_force.second.squaredNorm() *
                            weight_f_Cb);
  }
}

std::unordered_map<Finger, std::pair<Eigen::Vector2d, Eigen::Vector2d>>
InstantaneousContactForceQP::GetContactForceResult(
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
    const BrickType brick_type, const multibody::MultibodyPlant<double>* plant,
    const Eigen::Ref<const Eigen::Matrix2d>& Kp_tr,
    const Eigen::Ref<const Eigen::Matrix2d>& Kd_tr, double Kp_ro, double Kd_ro,
    double weight_a_error, double weight_thetaddot_error, double weight_f_Cb_B,
    double mu, double translational_damping, double rotational_damping,
    double I_B, double mass_B)
    : brick_type_(brick_type),
      plant_{plant},
      mu_{mu},
      Kp_tr_{Kp_tr},
      Kd_tr_{Kd_tr},
      Kp_ro_{Kp_ro},
      Kd_ro_{Kd_ro},
      weight_a_error_{weight_a_error},
      weight_thetaddot_error_{weight_thetaddot_error},
      weight_f_Cb_B_{weight_f_Cb_B},
      translational_damping_(translational_damping),
      rotational_damping_(rotational_damping),
      mass_B_(mass_B),
      I_B_(I_B) {
  // TODO(rcory) Check positive definiteness of Kp_tr and Kd_tr
  DRAKE_DEMAND(Kp_ro_ >= 0);
  DRAKE_DEMAND(Kd_ro_ >= 0);
  DRAKE_DEMAND(weight_a_error_ >= 0);
  DRAKE_DEMAND(weight_thetaddot_error_ >= 0);
  DRAKE_DEMAND(weight_f_Cb_B_ >= 0);
  DRAKE_DEMAND(translational_damping_ >= 0);
  DRAKE_DEMAND(rotational_damping_ >= 0);
  DRAKE_DEMAND(mass_B_ >= 0);
  DRAKE_DEMAND(I_B_ >= 0);

  output_index_fingers_control_ =
      this->DeclareAbstractOutputPort(
              "fingers_control",
              &InstantaneousContactForceQPController::CalcFingersControl)
          .get_index();

  output_index_brick_control_ =
      this->DeclareAbstractOutputPort(
              "brick_control",
              &InstantaneousContactForceQPController::CalcBrickControl)
          .get_index();

  input_index_estimated_plant_state_ =
      this->DeclareInputPort("estimated_plant_state", systems::kVectorValued,
                             plant->num_positions() + plant->num_velocities())
          .get_index();

  // This input port contains an unordered map of Finger to a pair of BrickFace
  // and contact point location. Typically if the finger is not in contact it
  // shouldn't be included in this input map. However, the QP planner will
  // happily compute forces/moments for any finger provided in this input (even
  // if in reality it is not in contact), by assuming the fingertip position is
  // a valid contact point on the brick (because this QP planner knows nothing
  // about geometry).
  input_index_finger_face_assignments_ =
      this->DeclareAbstractInputPort(
              "finger_face_assignments",
              Value<std::unordered_map<
                  Finger, std::pair<BrickFace, Eigen::Vector2d>>>{})
          .get_index();

  if (brick_type == BrickType::PlanarBrick) {
    input_index_desired_brick_state_ =  // {y, z, theta, ydot, zdot, thetadot}
        this->DeclareInputPort("desired_brick_state", systems::kVectorValued, 6)
            .get_index();
    input_index_desired_brick_acceleration_ =  // {yddot, zddot, thetaddot}
        this->DeclareInputPort("desired_brick_accel", systems::kVectorValued, 3)
            .get_index();  // {ay, az, w_x}

    brick_translate_y_position_index_ =
        plant_->GetJointByName("brick_translate_y_joint").position_start();
    brick_translate_z_position_index_ =
        plant_->GetJointByName("brick_translate_z_joint").position_start();
  } else {                              // Pin Brick
    input_index_desired_brick_state_ =  // {theta, thetadot}
        this->DeclareInputPort("desired_brick_state", systems::kVectorValued, 2)
            .get_index();
    input_index_desired_brick_acceleration_ =  // thetaddot
        this->DeclareInputPort("desired_brick_accel", systems::kVectorValued, 1)
            .get_index();
  }
  brick_revolute_x_position_index_ =
      plant_->GetJointByName("brick_revolute_x_joint").position_start();
  brick_body_index_ = plant_->GetBodyByName("brick_link").index();
}

// The output of this method will be an unordered map of size equal to the
// number of fingers in contact (not necessarily 3) and is dynamic. The info
// for what fingers are in contact and where is contained in the input
// finger_face_assignments.
void InstantaneousContactForceQPController::CalcFingersControl(
    const systems::Context<double>& context,
    std::unordered_map<Finger,
                       multibody::ExternallyAppliedSpatialForce<double>>*
        fingers_control) const {
  fingers_control->clear();
  const Eigen::VectorBlock<const VectorX<double>> plant_state =
      get_input_port_estimated_state().Eval(context);
  const Eigen::VectorBlock<const VectorX<double>> desired_brick_state =
      get_input_port_desired_brick_state().Eval(context);
  const Eigen::VectorBlock<const VectorX<double>> desired_brick_acceleration =
      get_input_port_desired_brick_acceleration().Eval(context);
  const std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>
      finger_face_assignments =
          get_input_port_finger_face_assignments()
              .Eval<std::unordered_map<Finger,
                                       std::pair<BrickFace, Eigen::Vector2d>>>(
                  context);

  // If the map `finger_face_assignments` is empty, then there is nothing to
  // compute. Just return empty in this case.
  if (finger_face_assignments.empty()) {
    return;
  }

  Eigen::Vector2d p_WB_planned;
  Eigen::Vector2d v_WB_planned;
  Eigen::Vector2d a_WB_planned;
  double theta_planned;
  double thetadot_planned;
  double thetaddot_planned;
  Vector6<double> brick_state;

  // TODO(rcory) Use properly defined indices to extract the desired values,
  //  instead of hard-coding them below.
  if (brick_type_ == BrickType::PlanarBrick) {
    p_WB_planned = desired_brick_state.head<2>();
    v_WB_planned = desired_brick_state.segment<2>(3);
    a_WB_planned = desired_brick_acceleration.head<2>();
    theta_planned = desired_brick_state(2);
    thetadot_planned = desired_brick_state(5);
    thetaddot_planned = desired_brick_acceleration(2);

    brick_state << plant_state(brick_translate_y_position_index_),
        plant_state(brick_translate_z_position_index_),
        plant_state(brick_revolute_x_position_index_),
        plant_state(plant_->num_positions() +
                    brick_translate_y_position_index_),
        plant_state(plant_->num_positions() +
                    brick_translate_z_position_index_),
        plant_state(plant_->num_positions() + brick_revolute_x_position_index_);
  } else {  // brick_type is PinBrick
    p_WB_planned = Eigen::Vector2d::Zero();
    v_WB_planned = Eigen::Vector2d::Zero();
    a_WB_planned = Eigen::Vector2d::Zero();
    theta_planned = desired_brick_state(0);
    thetadot_planned = desired_brick_state(1);
    thetaddot_planned = desired_brick_acceleration(0);

    brick_state << 0, 0, plant_state(brick_revolute_x_position_index_), 0, 0,
        plant_state(plant_->num_positions() + brick_revolute_x_position_index_);
  }

  InstantaneousContactForceQP qp(
      brick_type_, p_WB_planned, v_WB_planned, a_WB_planned, theta_planned,
      thetadot_planned, thetaddot_planned, Kp_tr_, Kd_tr_, Kp_ro_, Kd_ro_,
      brick_state, finger_face_assignments, weight_a_error_,
      weight_thetaddot_error_, weight_f_Cb_B_, mu_, I_B_, mass_B_,
      rotational_damping_, translational_damping_);

  const auto qp_result = solvers::Solve(qp.prog());
  const std::unordered_map<Finger, std::pair<Eigen::Vector2d, Eigen::Vector2d>>
      finger_contact_result = qp.GetContactForceResult(qp_result);
  const double theta = brick_state(2);
  const double cos_theta = std::cos(theta);
  const double sin_theta = std::sin(theta);
  Eigen::Matrix2d R_WB;
  R_WB << cos_theta, -sin_theta, sin_theta, cos_theta;
  for (const auto& finger_face_assignment : finger_face_assignments) {
    multibody::ExternallyAppliedSpatialForce<double> spatial_force;
    spatial_force.body_index = brick_body_index_;
    const std::pair<Eigen::Vector2d, Eigen::Vector2d>& force_position =
        finger_contact_result.at(finger_face_assignment.first);
    //    drake::log()->info("force: \n{}", force_position.first);
    //    drake::log()->info("position: \n{}", force_position.second);
    const Eigen::Vector2d p_BCb = force_position.second;
    spatial_force.p_BoBq_B = Eigen::Vector3d(0, p_BCb(0), p_BCb(1));
    const Eigen::Vector2d f_Cb_W = R_WB * force_position.first;
    spatial_force.F_Bq_W = multibody::SpatialForce<double>(
        Eigen::Vector3d::Zero(), Eigen::Vector3d(0, f_Cb_W(0), f_Cb_W(1)));
    fingers_control->emplace(finger_face_assignment.first, spatial_force);
  }
}

void InstantaneousContactForceQPController::CalcBrickControl(
    const systems::Context<double>& context,
    std::vector<multibody::ExternallyAppliedSpatialForce<double>>*
        contact_forces) const {
  std::unordered_map<Finger, multibody::ExternallyAppliedSpatialForce<double>>
      fingers_control;
  CalcFingersControl(context, &fingers_control);
  contact_forces->clear();
  contact_forces->reserve(fingers_control.size());
  for (const auto& finger_control : fingers_control) {
    contact_forces->push_back(finger_control.second);
  }
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
