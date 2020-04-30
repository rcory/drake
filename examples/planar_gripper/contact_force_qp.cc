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
    const BrickType& brick_type,
    const Eigen::Ref<const VectorX<double>>& brick_state,
    const Eigen::Ref<const VectorX<double>>& brick_state_desired,
    const Eigen::Ref<const VectorX<double>>& brick_accel_feedforward,
    const Eigen::Ref<const Eigen::Matrix2d>& Kp_t,
    const Eigen::Ref<const Eigen::Matrix2d>& Kd_t, double Kp_r, double Kd_r,
    const std::unordered_map<Finger, BrickFaceInfo>& finger_face_assignments,
    double weight_a_error, double weight_thetaddot_error, double weight_f_Cb,
    double mu, double I_B, double brick_mass, double rotational_damping,
    double translational_damping)
    : prog_{new solvers::MathematicalProgram()} {
  Eigen::Vector2d p_WB = Eigen::Vector2d::Zero();
  Eigen::Vector2d p_WB_desired = Eigen::Vector2d::Zero();
  Eigen::Vector2d v_WB = Eigen::Vector2d::Zero();
  Eigen::Vector2d v_WB_desired = Eigen::Vector2d::Zero();
  Eigen::Vector2d a_WB_feedforward = Eigen::Vector2d::Zero();
  double theta, thetadot;
  double theta_desired, thetadot_desired, thetaddot_feedforward;

  if (brick_type == BrickType::PinBrick) {
    // Extract the pined brick state: given as [θ, θ̇̇ ].
    theta = brick_state(0);
    thetadot = brick_state(1);

    // Extract the desired values.
    theta_desired = brick_state_desired(0);
    thetadot_desired = brick_state_desired(1);
    thetaddot_feedforward = brick_accel_feedforward(0);
  } else if (brick_type == BrickType::PlanarBrick) {
    // Extract the planar brick position and velocity values from the incoming
    // planar brick state vector, given as [y, z, θ, ẏ, ż, θ̇ ̇̇]ᵀ
    p_WB = brick_state.head<2>();
    v_WB = brick_state.segment<2>(3);
    theta = brick_state(2);
    thetadot = brick_state(5);

    // Extract the desired values.
    p_WB_desired = brick_state_desired.head<2>();
    v_WB_desired = brick_state_desired.segment<2>(3);
    theta_desired = brick_state_desired(2);
    thetadot_desired = brick_state_desired(5);
    a_WB_feedforward = brick_accel_feedforward.head<2>();
    thetaddot_feedforward = brick_accel_feedforward(2);
  } else {
    throw std::logic_error("Unknown BrickType.");
  }

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
        GetFrictionConeEdges(mu, finger_face_assignment.second.brick_face);
    // Compute the finger contact position Cbi
    Vector2<double> p_BCbi = finger_face_assignment.second.p_BCb;
    Vector2<symbolic::Expression> f_Cbi_B =
        friction_cone_edges_B * f_Cbi_B_edges;
    f_Cb_B.emplace(finger_face_assignment.first, f_Cbi_B);
    total_torque += p_BCbi(0) * f_Cbi_B(1) - p_BCbi(1) * f_Cbi_B(0);
    finger_face_contacts_.emplace_back(finger_face_assignment.first,
                                       finger_face_assignment.second.brick_face,
                                       f_Cbi_B_edges, mu, p_BCbi);
  }

  // Add the rotational damping term
  total_torque += -rotational_damping * thetadot;

  // Form the desired angular acceleration as a PD + FF term.
  const double thetaddot_des = Kp_r * (theta_desired - theta) +
                               Kd_r * (thetadot_desired - thetadot) +
                               thetaddot_feedforward;
  symbolic::Expression thetaddot = total_torque / I_B;
  prog_->AddQuadraticCost(pow(thetaddot - thetaddot_des, 2) *
                          weight_thetaddot_error);

  // If applicable, add cost on linear acceleration.
  if (brick_type == BrickType::PlanarBrick) {
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
            (R_WB * f_total_B - D * v_WB) / brick_mass;
    const Eigen::Vector2d a_WB_des = Kp_t * (p_WB_desired - p_WB) +
        Kd_t * (v_WB_desired - v_WB) + a_WB_feedforward;
    prog_->AddQuadraticCost((a_WB - a_WB_des).squaredNorm() * weight_a_error);
  }

  // Adds a cost on the squared norm of the forces.
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
    const Eigen::Ref<const Eigen::Matrix2d>& Kp_t,
    const Eigen::Ref<const Eigen::Matrix2d>& Kd_t, double Kp_r, double Kd_r,
    double weight_a_error, double weight_thetaddot_error, double weight_f_Cb_B,
    double mu, double translational_damping, double rotational_damping,
    double I_B, double mass_B)
    : brick_type_(brick_type),
      plant_{plant},
      mu_{mu},
      Kp_t_{Kp_t},
      Kd_t_{Kd_t},
      Kp_r_{Kp_r},
      Kd_r_{Kd_r},
      weight_a_error_{weight_a_error},
      weight_thetaddot_error_{weight_thetaddot_error},
      weight_f_Cb_B_{weight_f_Cb_B},
      translational_damping_(translational_damping),
      rotational_damping_(rotational_damping),
      mass_B_(mass_B),
      I_B_(I_B) {
  // TODO(rcory) Check positive definiteness of Kp_tr and Kd_tr
  DRAKE_DEMAND(Kp_r_ >= 0);
  DRAKE_DEMAND(Kd_r_ >= 0);
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

  // This input port contains an unordered map of Finger to BrickFaceInfo. The
  // QP controller will determine which fingers to include in the contact
  // force/moment calculations by examining the `is_in_contact` boolean flag of
  // the BrickFaceInfo struct. If this flag is set to false, the corresponding
  // finger will *not* be included in any of the force/moment calculations.
  input_index_finger_face_assignments_ =
      this->DeclareAbstractInputPort(
              "finger_face_assignments",
              Value<std::unordered_map<Finger, BrickFaceInfo>>{})
          .get_index();

  if (brick_type == BrickType::PlanarBrick) {
    input_index_desired_brick_state_ =
        this->DeclareInputPort("desired_brick_state", systems::kVectorValued, 6)
            .get_index();
    input_index_desired_brick_acceleration_ =
        this->DeclareInputPort("desired_brick_accel", systems::kVectorValued, 3)
            .get_index();
    brick_translate_y_position_index_ =
        plant_->GetJointByName("brick_translate_y_joint").position_start();
    brick_translate_z_position_index_ =
        plant_->GetJointByName("brick_translate_z_joint").position_start();
  } else {  // Pin Brick
    input_index_desired_brick_state_ =
        this->DeclareInputPort("desired_brick_state", systems::kVectorValued, 2)
            .get_index();
    input_index_desired_brick_acceleration_ =
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
  const std::unordered_map<Finger, BrickFaceInfo> finger_face_assignments_in =
      get_input_port_finger_face_assignments()
          .Eval<std::unordered_map<Finger, BrickFaceInfo>>(context);

  // Determine which fingers to actually include in the QP optimization. If the
  // finger is in contact, then it is included in the calculations. If it is not
  // in contact, it is ignored.
  std::unordered_map<Finger, BrickFaceInfo> finger_face_assignments;
  for (auto iter : finger_face_assignments_in) {
    if (iter.second.is_in_contact) {
      finger_face_assignments.emplace(iter.first, iter.second);
    }
  }

  // If the map `finger_face_assignments` is empty, then there is nothing to
  // compute. Just return empty in this case.
  if (finger_face_assignments.empty()) {
    return;
  }

  VectorX<double> brick_state;
  if (brick_type_ == BrickType::PlanarBrick) {
    brick_state = Vector6<double>::Zero();
    brick_state << plant_state(brick_translate_y_position_index_),
        plant_state(brick_translate_z_position_index_),
        plant_state(brick_revolute_x_position_index_),
        plant_state(plant_->num_positions() +
                    brick_translate_y_position_index_),
        plant_state(plant_->num_positions() +
                    brick_translate_z_position_index_),
        plant_state(plant_->num_positions() + brick_revolute_x_position_index_);
  } else {  // brick_type is PinBrick
    brick_state = Eigen::Vector2d::Zero();
    brick_state << plant_state(brick_revolute_x_position_index_),
        plant_state(plant_->num_positions() + brick_revolute_x_position_index_);
  }

  InstantaneousContactForceQP qp(
      brick_type_, brick_state, desired_brick_state, desired_brick_acceleration,
      Kp_t_, Kd_t_, Kp_r_, Kd_r_, finger_face_assignments, weight_a_error_,
      weight_thetaddot_error_, weight_f_Cb_B_, mu_, I_B_, mass_B_,
      rotational_damping_, translational_damping_);

  const auto qp_result = solvers::Solve(qp.prog());
  const std::unordered_map<Finger, std::pair<Eigen::Vector2d, Eigen::Vector2d>>
      finger_contact_result = qp.GetContactForceResult(qp_result);
  const double theta = plant_state(brick_revolute_x_position_index_);
  const double cos_theta = std::cos(theta);
  const double sin_theta = std::sin(theta);
  Eigen::Matrix2d R_WB;
  R_WB << cos_theta, -sin_theta, sin_theta, cos_theta;
  for (const auto& finger_face_assignment : finger_face_assignments) {
    multibody::ExternallyAppliedSpatialForce<double> spatial_force;
    spatial_force.body_index = brick_body_index_;
    const std::pair<Eigen::Vector2d, Eigen::Vector2d>& force_position =
        finger_contact_result.at(finger_face_assignment.first);
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
