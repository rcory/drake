#include "drake/examples/planar_gripper/contact_force_qp.h"

#include <limits>

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
    const systems::Context<double>& plant_context, double weight_a,
    double weight_thetaddot, double weight_f_Cb,
    const std::map<Finger, BrickFace>& finger_face_assignment)
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
    finger_face_contacts_.emplace_back(finger_face.first, finger_face.second,
                                       f_Cbi_B_edges, mu);
    // Compute the finger contact position Cbi
    Vector3<double> p_BFingertip;
    gripper_brick_->plant().CalcPointsPositions(
        plant_context, gripper_brick->finger_link2_frame(finger_face.first),
        gripper_brick_->p_L2Fingertip(), gripper_brick_->brick_frame(),
        &p_BFingertip);
    // Now compute the edges of the friction cone in the brick frame. The edges
    // only depends on the brick face.
    const Eigen::Matrix2d friction_cone_edges_B =
        GetFrictionConeEdges(mu, finger_face.second);
    Vector2<double> p_BCbi = p_BFingertip.tail<2>();
    switch (finger_face.second) {
      case BrickFace::kNegY: {
        p_BCbi(0) += gripper_brick_->finger_tip_radius();
        break;
      }
      case BrickFace::kPosY: {
        p_BCbi(0) -= gripper_brick_->finger_tip_radius();
        break;
      }
      case BrickFace::kNegZ: {
        p_BCbi(1) += gripper_brick_->finger_tip_radius();
        break;
      }
      case BrickFace::kPosZ: {
        p_BCbi(1) -= gripper_brick_->finger_tip_radius();
        break;
      }
      default: { throw std::runtime_error("Unknown face."); }
    }
    Vector2<symbolic::Expression> f_Cbi_B =
        friction_cone_edges_B * f_Cbi_B_edges;
    f_Cb_B.emplace(finger_face.first, f_Cbi_B);
    total_torque += p_BCbi(0) * f_Cbi_B(1) - p_BCbi(1) * f_Cbi_B(0);
  }
  const VectorX<double> q = gripper_brick_->plant().GetPositions(plant_context);
  const VectorX<double> v =
      gripper_brick_->plant().GetVelocities(plant_context);
  // First compute the desired acceleration.
  const Eigen::Vector2d p_WB(
      q(gripper_brick_->brick_translate_y_position_index()),
      q(gripper_brick_->brick_translate_z_position_index()));
  const Eigen::Vector2d v_WB(
      v(gripper_brick_->brick_translate_y_position_index()),
      v(gripper_brick_->brick_translate_z_position_index()));
  const Eigen::Vector2d a_WB_des =
      Kp1 * (p_WB_planned - p_WB) + Kd1 * (v_WB_planned - v_WB) + a_WB_planned;
  const double theta = q(gripper_brick_->brick_revolute_x_position_index());
  const double thetadot = v(gripper_brick_->brick_revolute_x_position_index());
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
          0, -multibody::UniformGravityFieldElement<double>::kDefaultStrength) +
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

std::unordered_map<Finger, Eigen::Vector2d>
InstantaneousContactForceQP::GetFingerContactForceResult(
    const solvers::MathematicalProgramResult& result) const {
  std::unordered_map<Finger, Eigen::Vector2d> finger_contact_forces;
  finger_contact_forces.reserve(finger_face_contacts_.size());
  for (const auto& finger_face_contact : finger_face_contacts_) {
    const Eigen::Vector2d f_Cbi_B_edges_sol =
        result.GetSolution(finger_face_contact.f_Cb_B_edges);
    const Eigen::Matrix2d friction_cone_edges_B = GetFrictionConeEdges(
        finger_face_contact.mu, finger_face_contact.brick_face);
    finger_contact_forces.emplace(finger_face_contact.finger,
                                  friction_cone_edges_B * f_Cbi_B_edges_sol);
  }
  return finger_contact_forces;
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
