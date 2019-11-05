#include "drake/examples/planar_gripper/planar_finger_qp.h"

#include <limits>

#include "drake/solvers/solve.h"

namespace drake {
namespace examples {
namespace planar_gripper {
const double kInf = std::numeric_limits<double>::infinity();

PlanarFingerInstantaneousQP::PlanarFingerInstantaneousQP(
    const multibody::MultibodyPlant<double>* finger_brick, double theta_planned,
    double thetadot_planned, double thetaddot_planned, double Kp, double Kd,
    double theta, double thetadot,
    const Eigen::Ref<const Eigen::Vector2d>& p_BFingerTip,
    double weight_thetaddot_error, double weight_f_Cb, BrickFace contact_face,
    double mu, double I_B, double finger_tip_radius, double damping)
    : plant_{finger_brick},
      prog_{new solvers::MathematicalProgram()},
      f_Cb_B_edges_{prog_->NewContinuousVariables<2>()} {
  prog_->AddBoundingBoxConstraint(0, kInf, f_Cb_B_edges_);
  p_BCb_ = p_BFingerTip;
  switch (contact_face) {
    case BrickFace::kPosZ: {
      friction_cone_edges_ << -mu, mu, -1, -1;
      break;
    }
    case BrickFace::kNegZ: {
      friction_cone_edges_ << -mu, mu, 1, 1;
      break;
    }
    case BrickFace::kPosY: {
      friction_cone_edges_ << -1, -1, -mu, mu;
      break;
    }
    case BrickFace::kNegY: {
      friction_cone_edges_ << 1, 1, -mu, mu;
      break;
    }
    default: { throw std::runtime_error("Unknown face."); }
  }
  unused(finger_tip_radius);
  Vector2<symbolic::Expression> f_Cb_B = friction_cone_edges_ * f_Cb_B_edges_;
  // Now compute thetaddot
  const symbolic::Expression thetaddot =
      (p_BCb_(0) * f_Cb_B(1) - p_BCb_(1) * f_Cb_B(0) - damping * thetadot) / I_B;

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

PlanarFingerInstantaneousQPController::PlanarFingerInstantaneousQPController(
    const multibody::MultibodyPlant<double>* plant, double Kp, double Kd,
    double weight_thetaddot, double weight_f_Cb_B, double mu,
    double finger_tip_radius, double damping)
    : plant_{plant},
      mu_{mu},
      Kp_{Kp},
      Kd_{Kd},
      weight_thetaddot_{weight_thetaddot},
      weight_f_Cb_B_{weight_f_Cb_B},
      finger_tip_radius_{finger_tip_radius},
      damping_(damping) {
  DRAKE_DEMAND(Kp_ >= 0);
  DRAKE_DEMAND(Kd_ >= 0);
  DRAKE_DEMAND(weight_thetaddot_ >= 0);
  DRAKE_DEMAND(weight_f_Cb_B_ >= 0);

  I_B_ = dynamic_cast<const multibody::RigidBody<double>&>(
             plant_->GetFrameByName("brick_base_link").body())
             .default_rotational_inertia()
             .get_moments()(0);

  output_index_control_ =
      this->DeclareAbstractOutputPort(
              "spatial_force",
              &PlanarFingerInstantaneousQPController::CalcControl)
          .get_index();

  input_index_state_ =
      this->DeclareInputPort(systems::kVectorValued,
                             plant_->num_positions() + plant_->num_velocities())
          .get_index();
  input_index_desired_brick_state_ =
      this->DeclareInputPort(systems::kVectorValued, 2).get_index();
  input_index_desired_thetaddot_ =
      this->DeclareInputPort(systems::kVectorValued, 1).get_index();
  input_index_p_BFingerTip_ =
      this->DeclareInputPort(systems::kVectorValued, 2).get_index();
  input_index_contact_face_ =
      this->DeclareAbstractInputPort("contact_face", Value<BrickFace>{})
          .get_index();

  brick_revolute_position_index_ =
      plant_->GetJointByName("brick_pin_joint").position_start();
  brick_body_index_ = plant_->GetBodyByName("brick_base_link").index();
}

void PlanarFingerInstantaneousQPController::CalcControl(
    const systems::Context<double>& context,
    std::vector<multibody::ExternallyAppliedSpatialForce<double>>* control)
    const {
  const Eigen::VectorBlock<const VectorX<double>> state =
      get_input_port_estimated_state().Eval(context);
  const Eigen::VectorBlock<const VectorX<double>> brick_state_desired =
      get_input_port_desired_state().Eval(context);
  const Eigen::VectorBlock<const VectorX<double>> thetaddot_planned =
      get_input_port_desired_thetaddot().Eval(context);
  const Eigen::VectorBlock<const VectorX<double>> p_BFingerTip =
      get_input_port_p_BFingerTip().Eval(context);
  const BrickFace contact_face =
      get_input_port_contact_face().Eval<BrickFace>(context);
  const double theta = state(brick_revolute_position_index_);
  const double thetadot =
      state(plant_->num_positions() + brick_revolute_position_index_);

          PlanarFingerInstantaneousQP qp(
              plant_, brick_state_desired(0), brick_state_desired(1),
              thetaddot_planned(0), Kp_, Kd_, theta, thetadot, p_BFingerTip,
              weight_thetaddot_, weight_f_Cb_B_, contact_face, mu_, I_B_,
              finger_tip_radius_, damping_);

  const auto qp_result = solvers::Solve(qp.prog());

  const Vector2<double> f_Cb_B = qp.GetContactForceResult(qp_result);
  const double cos_theta = std::cos(theta);
  const double sin_theta = std::sin(theta);
  Eigen::Matrix2d R_WB;
  R_WB << cos_theta, -sin_theta, sin_theta, cos_theta;
  Eigen::Vector2d f_Cb_W = R_WB * f_Cb_B;
  control->resize(1);
  (*control)[0].body_index = brick_body_index_;
  (*control)[0].p_BoBq_B = Eigen::Vector3d(0, qp.p_BCb()(0), qp.p_BCb()(1));
  (*control)[0].F_Bq_W = multibody::SpatialForce<double>(
      Eigen::Vector3d::Zero(), Eigen::Vector3d(0, f_Cb_W(0), f_Cb_W(1)));
//   drake::log()->info("f_Cb_W: \n{}", f_Cb_W);
//   unused(f_Cb_W);
//   (*control)[0].F_Bq_W = multibody::SpatialForce<double>(
//       Eigen::Vector3d::Zero(), Eigen::Vector3d(0, -0.037, -0.029));
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
