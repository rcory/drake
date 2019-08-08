#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/examples/planar_gripper/gripper_brick.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"

namespace drake {
namespace examples {
namespace planar_gripper {
struct FingerTransition {
  FingerTransition(int m_start_knot_index, int m_end_knot_index,
                   Finger m_finger, BrickFace m_to_face)
      : start_knot_index(m_start_knot_index),
        end_knot_index(m_end_knot_index),
        finger(m_finger),
        to_face(m_to_face) {}
  int start_knot_index;
  int end_knot_index;
  Finger finger;
  BrickFace to_face;
};

/**
 * Given a contact mode sequence, find the finger joint trajectory / brick
 * trajectory and contact forces, such that the object is reoriented.
 */
class GripperBrickTrajectoryOptimization {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GripperBrickTrajectoryOptimization);

  enum class IntegrationMethod {
    kBackwardEuler,
    kMidpoint,
  };

  struct Options {
    Options(double m_face_shrink_factor, IntegrationMethod m_integration_method,
            double m_rolling_angle_bound, double m_collision_avoidance_margin,
            double m_depth, double m_friction_cone_shrink_factor)
        : face_shrink_factor(m_face_shrink_factor),
          integration_method{m_integration_method},
          rolling_angle_bound{m_rolling_angle_bound},
          collision_avoidance_margin{m_collision_avoidance_margin},
          depth{m_depth},
          friction_cone_shrink_factor{m_friction_cone_shrink_factor} {}
    double face_shrink_factor = 0.8;
    IntegrationMethod integration_method = IntegrationMethod::kBackwardEuler;
    double rolling_angle_bound = {0.05 * M_PI};
    double collision_avoidance_margin = 0.01;
    double depth = 0.001;
    double friction_cone_shrink_factor = 0.9;
  };

  /**
   * @param gripper_brick The system for which to plan the trajectory.
   * @param nT The number of knot points.
   * @param initial_contact The assignment of finger to brick face at the
   * initial state.
   * @param finger_transitions All the finger transitions in the trajectory.
   * @note we allow at most one finger in transition at each knot, and each
   * transition must takes at least two time intervals (namely end_knot_index -
   * start_knot_index >= 2).
   */
  GripperBrickTrajectoryOptimization(
      const GripperBrickHelper<double>* gripper_brick, int nT,
      const std::map<Finger, BrickFace>& initial_contact,
      const std::vector<FingerTransition>& finger_transitions,
      double brick_lid_friction_force_magnitude,
      double brick_lid_friction_torque_magnitude, const Options& options);

  const solvers::MathematicalProgram& prog() const { return *prog_; }

  solvers::MathematicalProgram* get_mutable_prog() { return prog_.get(); }

  const std::vector<std::map<Finger, BrickFace>>& finger_face_contacts() const {
    return finger_face_contacts_;
  }

  const VectorX<symbolic::Variable>& dt() const { return dt_; }

  const MatrixX<symbolic::Variable>& q_vars() const { return q_vars_; }

  const VectorX<symbolic::Variable>& brick_v_y_vars() const {
    return brick_v_y_vars_;
  }

  const VectorX<symbolic::Variable>& brick_v_z_vars() const {
    return brick_v_z_vars_;
  }

  const VectorX<symbolic::Variable>& brick_omega_x_vars() const {
    return brick_omega_x_vars_;
  }

  const std::vector<std::unordered_map<Finger, Vector2<symbolic::Variable>>>&
  f_FB_B() const {
    return f_FB_B_;
  }

  systems::Context<double>* plant_mutable_context(int knot) {
    return plant_mutable_contexts_[knot];
  }

  /**
   * Add a bound on the difference of an entry in q between two consecutive
   * knot. Namely |q(position_index, left_knot + 1) - q(position_index,
   * left_knot)| <= bound.
   */
  void AddPositionDifferenceBound(int left_knot, int position_index,
                                  double bound);

  /**
   * Constrain that the linearly interpolated posture (1-λ)*q[i] + λ*q[i+1] is
   * collision free for a given pair of geometries.
   * @param left_knot The posture is a linear interpolation of q[left_knot] and
   * q[left_knot + 1].
   * @param fraction λ in the documentation above. @throw invalid argument if
   * λ>=1 or λ<=0.
   * @param geometry_pair The pair of geometries that we want to be separated.
   * @param minimal_distance The lower bound on the separation distance.
   */
  void AddCollisionAvoidanceForInterpolatedPosture(
      int left_knot, double fraction,
      const SortedPair<geometry::GeometryId>& geometry_pair,
      double minimal_distance);

  /**
   * Constrain that at a given knot, the brick is in static equilibrium.
   */
  void AddBrickStaticEquilibriumConstraint(int knot);

  trajectories::PiecewisePolynomial<double> ReconstructFingerTrajectory(
      const solvers::MathematicalProgramResult& result,
      const std::map<std::string, int>& finger_joint_name_to_row_index_map)
      const;

  /**
   * Returns the contacting face of a given finger at a given time. If the
   * finger is not in contact, then return empty.
   */
  optional<BrickFace> GetFingerContactFace(
      double t, Finger finger,
      const Eigen::Ref<const Eigen::VectorXd>& t_sol) const;

  /**
   * Reconstruct the finger force trajectory (i.e., the contact force applied at
   * the finger tip contact location, expressed in the brick frame). Notice that
   * this force is applied at the finger tip contact location, not at the finger
   * tip sphere center,  not at the finger link2 origin.
   */
  trajectories::PiecewisePolynomial<double> ReconstructFingerForceTrajectory(
      Finger finger, const solvers::MathematicalProgramResult& result) const;

  Eigen::VectorXd ReconstructTimeSolution(
      const solvers::MathematicalProgramResult& result) const;

  /**
   * For a given finger and a given time, return the tangential position of
   * the finger/brick contact location in the solution. If the finger is not in
   * contact, then returns empty.
   * Note that we only compute the tangential position. Namely if the contact is
   * on +y or -y face of the brick, then we return the z coordinate of the
   * contact point in the brick frame; if the contact is on +z or -z face of the
   * brick, then we return the y coordinate of the contact point in the brick
   * frame.
   * @param t The query time.
   * @param finger The query finger.
   * @param t_sol The result from calling ReconstructTimeSolution().
   * @param q_sol The result from calling prog().GetSolution(q_vars()).
   * @param plant_mutable_context The context for the plant, please create it
   * by doing diagram.CreateDefaultContext();
   * diagram.GetMutableSubsystemContext();
   */
  optional<std::pair<double, BrickFace>> GetFingerContactLocationOnBrickFace(
      double t, Finger finger, const Eigen::Ref<const Eigen::VectorXd>& t_sol,
      const Eigen::Ref<const Eigen::MatrixXd>& q_sol,
      systems::Context<double>* plant_mutable_context) const;

 private:
  void AssignVariableForContactForces(
      const std::map<Finger, BrickFace>& initial_contact,
      std::vector<const FingerTransition*>* sorted_finger_transitions);

  void AddDynamicConstraint(
      const std::vector<const FingerTransition*>& sorted_finger_transitions,
      double brick_lid_friction_force_magnitude,
      double brick_lid_friction_torque_magnitude,
      IntegrationMethod integration_method);

  // Add midpoint integration constraint on the brick position.
  void AddBrickPositionIntegrationConstraint();

  // For specified (finger/face) contact pairs, they have to satisfy the
  // kinematic constraint that the finger touches the face, and does not slide
  // on the face.
  void AddKinematicInContactConstraint(double face_shrink_factor,
                                       double rolling_angle_bound,
                                       double depth);

  void AddCollisionAvoidanceConstraint(
      const std::vector<const FingerTransition*>& sorted_finger_transitions,
      double collision_avoidance_margin);

  void AddFrictionConeConstraints(double friction_cone_shrink_factor);

  void AddMiddlePointIntegrationConstraint();

  const GripperBrickHelper<double>* const gripper_brick_;
  // number of knots.
  int nT_;
  std::unique_ptr<solvers::MathematicalProgram> prog_;
  // q_vars_.col(i) represents all the q variables at the i'th knot.
  MatrixX<symbolic::Variable> q_vars_;

  MatrixX<symbolic::Variable> v_vars_;

  // brick_v_y_vars_(i) represents the brick y translational velocity variable
  // at the i'th knot.
  VectorX<symbolic::Variable> brick_v_y_vars_;
  // brick_v_z_vars_(i) represents the brick z translational velocity variable
  // at the i'th knot.
  VectorX<symbolic::Variable> brick_v_z_vars_;
  // brick_omega_x_vars_(i) represents the brick z translational velocity
  // variable at the i'th knot.
  VectorX<symbolic::Variable> brick_omega_x_vars_;
  // f_FB_B_[knot][finger] represents the contact force from the finger (F) to
  // the brick (B) expressed in the brick (B) frame at a given knot for a given
  // finger.
  std::vector<std::unordered_map<Finger, Vector2<symbolic::Variable>>> f_FB_B_;
  // diagram_contexts_[i] is the diagram context for the i'th knot.
  std::vector<std::unique_ptr<systems::Context<double>>> diagram_contexts_;
  std::vector<systems::Context<double>*> plant_mutable_contexts_;
  VectorX<symbolic::Variable> dt_;
  std::vector<FingerTransition> finger_transitions_;
  std::vector<std::map<Finger, BrickFace>> finger_face_contacts_;

  // We will also impose constraints on the mid point between two knots.
  MatrixX<symbolic::Variable> q_mid_vars_;
  std::vector<std::unique_ptr<systems::Context<double>>> diagram_contexts_mid_;
  std::vector<systems::Context<double>*> plant_mutable_contexts_mid_;

  // We will impose kinematic constraint on some interpolated postures, hence we
  // need to create the context for these postures, and keep the contexts alive
  // during optimization.
  std::vector<std::unique_ptr<systems::Context<double>>>
      diagram_contexts_interpolated_;
};
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
