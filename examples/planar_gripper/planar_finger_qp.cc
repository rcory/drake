#include "drake/examples/planar_gripper/planar_finger_qp.h"

#include <limits>

namespace drake {
namespace examples {
namespace planar_gripper {
const double kInf = std::numeric_limits<double>::infinity();

PlanarFingerInstantaneousQP::PlanarFingerInstantaneousQP(
    const multibody::MultibodyPlant<double>* finger_brick, double theta_planned,
    double thetadot_planned, double thetaddot_planned, double Kp, double Kd,
    const systems::Context<double>& plant_context,
    double weight_thetaddot_error, double weight_f_Cb, BrickFace contact_face,
    double mu, const Eigen::Vector3d& p_L2FingerTip, double I_B,
    double finger_tip_radius)
    : plant_{finger_brick},
      prog_{new solvers::MathematicalProgram()},
      f_Cb_B_edges_{prog_->NewContinuousVariables<2>()} {
  prog_->AddBoundingBoxConstraint(0, kInf, f_Cb_B_edges_);
  Eigen::Vector3d p_BFingerTip;
  plant_->CalcPointsPositions(
      plant_context, plant_->GetFrameByName("finger_link2"), p_L2FingerTip,
      plant_->GetFrameByName("brick_base_link"), &p_BFingerTip);
  Eigen::Vector2d p_BCb = p_BFingerTip.tail<2>();
  switch (contact_face) {
    case BrickFace::kPosZ: {
      friction_cone_edges_ << -mu, mu, -1, -1;
      p_BCb(1) -= finger_tip_radius;
      break;
    }
    case BrickFace::kNegZ: {
      friction_cone_edges_ << -mu, mu, 1, 1;
      p_BCb(1) += finger_tip_radius;
      break;
    }
    case BrickFace::kPosY: {
      friction_cone_edges_ << -1, -1, -mu, mu;
      p_BCb(0) -= finger_tip_radius;
      break;
    }
    case BrickFace::kNegY: {
      friction_cone_edges_ << 1, 1, -mu, mu;
      p_BCb(0) += finger_tip_radius;
      break;
    }
    default: { throw std::runtime_error("Unknown face."); }
  }
  Vector2<symbolic::Expression> f_Cb_B = friction_cone_edges_ * f_Cb_B_edges_;
  // Now compute thetaddot
  const symbolic::Expression thetaddot =
      (p_BCb(0) * f_Cb_B(1) - p_BCb(1) * f_Cb_B(0)) / I_B;

  const Eigen::Vector3d q = plant_->GetPositions(plant_context);
  const Eigen::Vector3d v = plant_->GetVelocities(plant_context);
  const int brick_theta_index =
      plant_->GetJointByName("brick_pin_joint").position_start();
  const double theta = q(brick_theta_index);
  const double thetadot = v(brick_theta_index);
  const double thetaddot_des = Kp * (theta_planned - theta) +
                               Kd * (thetadot_planned - thetadot) +
                               thetaddot_planned;
  prog_->AddQuadraticCost(weight_thetaddot_error * (thetaddot - thetaddot_des) *
                              (thetaddot - thetaddot_des) +
                          weight_f_Cb * f_Cb_B.squaredNorm());
}

const Eigen::Vector2d PlanarFingerInstantaneousQP::GetContactForceResult(
    const solvers::MathematicalProgramResult& result) const {
  return friction_cone_edges_ * result.GetSolution(f_Cb_B_edges_);
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
