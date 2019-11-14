#include "drake/examples/planar_gripper/brick_qp.h"

#include "drake/solvers/solve.h"

namespace drake {
namespace examples {
namespace planar_gripper {
const double kInf = std::numeric_limits<double>::infinity();

PlanarBrickInstantaneousQP::PlanarBrickInstantaneousQP(
    const multibody::MultibodyPlant<double>* brick, double theta_planned,
    double thetadot_planned, double thetaddot_planned, double Kp, double Kd,
    double theta, double thetadot, double weight_thetaddot_error,
    double weight_f_Cb, BrickFace contact_face,
    const Eigen::Ref<const Eigen::Vector2d>& p_BCb, double mu, double I_B,
    double damping)
    : plant_{brick},
      prog_{new solvers::MathematicalProgram()},
      f_Cb_B_edges_{prog_->NewContinuousVariables<2>()} {
  prog_->AddBoundingBoxConstraint(0, kInf, f_Cb_B_edges_);
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
  Vector2<symbolic::Expression> f_Cb_B = friction_cone_edges_ * f_Cb_B_edges_;
  // Now compute thetaddot
  const symbolic::Expression thetaddot =
      (p_BCb(0) * f_Cb_B(1) - p_BCb(1) * f_Cb_B(0) - damping * thetadot) / I_B;

  const double thetaddot_des = Kp * (theta_planned - theta) +
                               Kd * (thetadot_planned - thetadot) +
                               thetaddot_planned;
  prog_->AddQuadraticCost(weight_thetaddot_error * (thetaddot - thetaddot_des) *
                              (thetaddot - thetaddot_des) +
                          weight_f_Cb * f_Cb_B.squaredNorm());
}

const Eigen::Vector2d PlanarBrickInstantaneousQP::GetContactForceResult(
    const solvers::MathematicalProgramResult& result) const {
  return friction_cone_edges_ * result.GetSolution(f_Cb_B_edges_);
}

BrickInstantaneousQPController::BrickInstantaneousQPController(
    const multibody::MultibodyPlant<double>* plant, double Kp, double Kd,
    double weight_thetaddot, double weight_f_Cb_B, double mu, double damping,
    double I_B)
    : systems::LeafSystem<double>(),
      brick_{plant},
      mu_{mu},
      Kp_{Kp},
      Kd_{Kd},
      weight_thetaddot_{weight_thetaddot},
      weight_f_Cb_B_{weight_f_Cb_B},
      damping_(damping),
      I_B_(I_B) {
  DRAKE_DEMAND(Kp_ >= 0);
  DRAKE_DEMAND(Kd_ >= 0);
  DRAKE_DEMAND(weight_thetaddot_ >= 0);
  DRAKE_DEMAND(weight_f_Cb_B_ >= 0);

  output_index_control_ =
      this->DeclareAbstractOutputPort(
              "spatial_force", &BrickInstantaneousQPController::CalcControl)
          .get_index();

  input_index_state_ =
      this->DeclareInputPort(systems::kVectorValued, 2).get_index();

  input_index_desired_state_ =
      this->DeclareInputPort(systems::kVectorValued, 2).get_index();

  input_index_desired_acceleration_ =
      this->DeclareInputPort(systems::kVectorValued, 1).get_index();

  input_index_p_BCb_ =
      this->DeclareInputPort(systems::kVectorValued, 2).get_index();

  input_index_contact_face_ =
      this->DeclareAbstractInputPort("contact_face", Value<BrickFace>{})
          .get_index();
}

void BrickInstantaneousQPController::CalcControl(
    const systems::Context<double>& context,
    std::vector<multibody::ExternallyAppliedSpatialForce<double>>* control)
    const {
  const Eigen::VectorBlock<const VectorX<double>> state =
      get_input_port_estimated_state().Eval(context);
  const Eigen::VectorBlock<const VectorX<double>> state_d =
      get_input_port_desired_state().Eval(context);
  const Eigen::VectorBlock<const VectorX<double>> thetaddot_planned =
      get_input_port_desired_acceleration().Eval(context);
  const Eigen::VectorBlock<const VectorX<double>> p_BCb =
      get_input_port_p_BCb().Eval(context);
  const BrickFace contact_face =
      get_input_port_contact_face().Eval<BrickFace>(context);

  PlanarBrickInstantaneousQP qp(brick_, state_d(0), state_d(1),
                                thetaddot_planned(0), Kp_, Kd_, state(0),
                                state(1), weight_thetaddot_, weight_f_Cb_B_,
                                contact_face, p_BCb, mu_, I_B_, damping_);
  const auto qp_result = solvers::Solve(qp.prog());
  const Vector2<double> f_Cb_B = qp.GetContactForceResult(qp_result);
  const double cos_theta = std::cos(state(0));
  const double sin_theta = std::sin(state(0));
  Eigen::Matrix2d R_WB;
  R_WB << cos_theta, -sin_theta, sin_theta, cos_theta;
  Eigen::Vector2d f_Cb_W = R_WB * f_Cb_B;

//  drake::log()->info("r x f: {}", p_BCb(0)*f_Cb_B(1) - p_BCb(1)*f_Cb_B(0));

  control->resize(1);
  (*control)[0].body_index = brick_->GetBodyByName("brick_base_link").index();
  (*control)[0].p_BoBq_B = Eigen::Vector3d(0, p_BCb(0), p_BCb(1));
  (*control)[0].F_Bq_W = multibody::SpatialForce<double>(
      Eigen::Vector3d::Zero(), Eigen::Vector3d(0, f_Cb_W(0), f_Cb_W(1)));
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
