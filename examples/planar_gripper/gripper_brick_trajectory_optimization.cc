#include "drake/examples/planar_gripper/gripper_brick_trajectory_optimization.h"

#include <limits>
#include <set>
#include <utility>

#include "drake/examples/planar_gripper/brick_dynamic_constraint.h"
#include "drake/examples/planar_gripper/brick_static_equilibrium_constraint.h"
#include "drake/examples/planar_gripper/gripper_brick_planning_constraint_helper.h"
#include "drake/multibody/inverse_kinematics/distance_constraint.h"
#include "drake/multibody/inverse_kinematics/kinematic_constraint_utilities.h"
#include "drake/multibody/inverse_kinematics/position_constraint.h"
#include "drake/systems/trajectory_optimization/integration_constraint.h"

namespace drake {
namespace examples {
namespace planar_gripper {
const double kInf = std::numeric_limits<double>::infinity();

GripperBrickTrajectoryOptimization::GripperBrickTrajectoryOptimization(
    const GripperBrickHelper<double>* const gripper_brick, int nT,
    const std::map<Finger, BrickFace>& initial_contact,
    const std::vector<FingerTransition>& finger_transitions,
    double brick_lid_friction_force_magnitude,
    double brick_lid_friction_torque_magnitude, const Options& options)
    : gripper_brick_{gripper_brick},
      nT_{nT},
      prog_{std::make_unique<solvers::MathematicalProgram>()},
      q_vars_{prog_->NewContinuousVariables(
          gripper_brick_->plant().num_positions(), nT_)},
      brick_v_y_vars_{prog_->NewContinuousVariables(nT_)},
      brick_v_z_vars_{prog_->NewContinuousVariables(nT_)},
      brick_omega_x_vars_{prog_->NewContinuousVariables(nT_)},
      f_FB_B_(nT_),
      plant_mutable_contexts_(nT_),
      dt_(prog_->NewContinuousVariables(nT_ - 1)),
      finger_transitions_(finger_transitions),
      finger_face_contacts_(nT_),
      q_mid_vars_{prog_->NewContinuousVariables(
          gripper_brick_->plant().num_positions(), nT_ - 1)},
      plant_mutable_contexts_mid_(nT_ - 1) {
  // Create the contexts used at each knot.
  diagram_contexts_.reserve(nT_);
  for (int i = 0; i < nT_; ++i) {
    diagram_contexts_.push_back(
        gripper_brick->diagram().CreateDefaultContext());
    plant_mutable_contexts_[i] =
        &(gripper_brick_->diagram().GetMutableSubsystemContext(
            gripper_brick_->plant(), diagram_contexts_[i].get()));
  }
  diagram_contexts_mid_.reserve(nT_ - 1);
  for (int i = 0; i < nT_ - 1; ++i) {
    diagram_contexts_mid_.push_back(
        gripper_brick->diagram().CreateDefaultContext());
    plant_mutable_contexts_mid_[i] =
        &(gripper_brick_->diagram().GetMutableSubsystemContext(
            gripper_brick_->plant(), diagram_contexts_mid_[i].get()));
  }

  // Add joint limits.
  for (int i = 0; i < nT_; ++i) {
    prog_->AddBoundingBoxConstraint(
        gripper_brick_->plant().GetPositionLowerLimits(),
        gripper_brick_->plant().GetPositionUpperLimits(), q_vars_.col(i));
  }

  std::vector<const FingerTransition*> sorted_finger_transitions;
  AssignVariableForContactForces(initial_contact, &sorted_finger_transitions);

  // add the brick position integration constraint.
  AddBrickPositionIntegrationConstraint();
  // Add the integration constraint on the brick velocity.
  AddDynamicConstraint(
      sorted_finger_transitions, brick_lid_friction_force_magnitude,
      brick_lid_friction_torque_magnitude, options.integration_method);

  // dt_[i] must be non-negative.
  prog_->AddBoundingBoxConstraint(0, kInf, dt_);

  // Add the kinematic in-contact constraint.
  AddKinematicInContactConstraint(options.face_shrink_factor,
                                  options.rolling_angle_bound, options.depth);

  // During finger transition, the transition finger should avoid the brick.
  AddCollisionAvoidanceConstraint(sorted_finger_transitions,
                                  options.collision_avoidance_margin);

  // Add friction cone constraint.
  AddFrictionConeConstraints(options.friction_cone_shrink_factor);
}

void GripperBrickTrajectoryOptimization::AssignVariableForContactForces(
    const std::map<Finger, BrickFace>& initial_contact,
    std::vector<const FingerTransition*>* sorted_finger_transitions) {
  // I assume initially the system is in static equilibrium.
  // sort the transitions based on their starting time.
  sorted_finger_transitions->reserve(finger_transitions_.size());
  for (const auto& finger_transition : finger_transitions_) {
    if (finger_transition.end_knot_index - finger_transition.start_knot_index <
        2) {
      throw std::invalid_argument(
          "GripperBrickTrajectoryOptimization: finger transition must take at "
          "least two time intervals.");
    }
    sorted_finger_transitions->push_back(&finger_transition);
  }
  std::sort(
      sorted_finger_transitions->begin(), sorted_finger_transitions->end(),
      [](const FingerTransition* transition1,
         const FingerTransition* transition2) {
        return transition1->start_knot_index < transition2->start_knot_index;
      });
  if ((*sorted_finger_transitions)[0]->start_knot_index <= 0) {
    throw std::invalid_argument(
        "GripperBrickTrajectoryOptimization: the initial transition cannot "
        "start with knot 0.");
  }
  int last_transition_end_knot = 0;
  finger_face_contacts_[0] = initial_contact;
  for (const auto& finger_transition : *sorted_finger_transitions) {
    // From the end of last transtion to the start of this transition.
    for (int i = last_transition_end_knot + 1;
         i <= finger_transition->start_knot_index; ++i) {
      finger_face_contacts_[i] =
          finger_face_contacts_[last_transition_end_knot];
    }
    // During this transition.
    finger_face_contacts_[finger_transition->start_knot_index + 1] =
        finger_face_contacts_[finger_transition->start_knot_index];
    auto it =
        finger_face_contacts_[finger_transition->start_knot_index + 1].find(
            finger_transition->finger);
    if (it ==
        finger_face_contacts_[finger_transition->start_knot_index + 1].end()) {
      throw std::invalid_argument(
          "GripperBrickTrajectoryOptimization: two transitions both move " +
          to_string(finger_transition->finger) + " at knot " +
          std::to_string(finger_transition->start_knot_index + 1));
    }
    finger_face_contacts_[finger_transition->start_knot_index + 1].erase(it);
    for (int i = finger_transition->start_knot_index + 2;
         i <= finger_transition->end_knot_index; ++i) {
      finger_face_contacts_[i] =
          finger_face_contacts_[finger_transition->start_knot_index + 1];
    }
    // End of the transition.
    finger_face_contacts_[finger_transition->end_knot_index].emplace(
        finger_transition->finger, finger_transition->to_face);
    last_transition_end_knot = finger_transition->end_knot_index;
  }
  // Set finger_face_contact_ after the final transition.
  for (int i = last_transition_end_knot; i < nT_; ++i) {
    finger_face_contacts_[i] = finger_face_contacts_[last_transition_end_knot];
  }

  // Now assign the decision variables for the contact forces.
  for (int i = 0; i < nT_; ++i) {
    for (const auto& finger_face : finger_face_contacts_[i]) {
      auto f = prog_->NewContinuousVariables<2>();
      f_FB_B_[i].emplace(finger_face.first, f);
    }
  }
}

void GripperBrickTrajectoryOptimization::AddDynamicConstraint(
    const std::vector<const FingerTransition*>& sorted_finger_transitions,
    double brick_lid_friction_force_magnitude,
    double brick_lid_friction_torque_magnitude,
    GripperBrickTrajectoryOptimization::IntegrationMethod integration_method) {
  // Given a set of (finger, face) contact pairs, form the decision variable
  // f_FB_B that represents the contact forces for the set.
  auto compose_contact_force_variables =
      [this](int knot_index,
             const std::map<Finger, BrickFace>& finger_face_contacts,
             Matrix2X<symbolic::Variable>* f_FB_B) {
        f_FB_B->resize(2, finger_face_contacts.size());
        int finger_face_contact_count = 0;
        for (const auto& finger_face_contact : finger_face_contacts) {
          auto it = f_FB_B_[knot_index].find(finger_face_contact.first);
          DRAKE_DEMAND(it != f_FB_B_[knot_index].end());
          f_FB_B->col(finger_face_contact_count++) = it->second;
        }
      };

  auto add_integration_constraint =
      [this, &integration_method, &brick_lid_friction_force_magnitude,
       &brick_lid_friction_torque_magnitude, &compose_contact_force_variables](
          int left_knot_index,
          const std::map<Finger, BrickFace>& finger_face_contacts) {
        Matrix2X<symbolic::Variable> f_FB_B_r(2, finger_face_contacts.size());
        const int right_knot_index = left_knot_index + 1;
        compose_contact_force_variables(right_knot_index, finger_face_contacts,
                                        &f_FB_B_r);
        Matrix2X<symbolic::Variable> f_FB_B_l(2, finger_face_contacts.size());
        compose_contact_force_variables(left_knot_index, finger_face_contacts,
                                        &f_FB_B_l);
        switch (integration_method) {
          case IntegrationMethod::kBackwardEuler: {
            auto constraint =
                std::make_shared<BrickDynamicBackwardEulerConstraint>(
                    gripper_brick_, plant_mutable_contexts_[right_knot_index],
                    finger_face_contacts, brick_lid_friction_force_magnitude,
                    brick_lid_friction_torque_magnitude);
            VectorX<symbolic::Variable> bound_vars;
            constraint->ComposeX<symbolic::Variable>(
                this->q_vars_.col(right_knot_index),
                this->brick_v_y_vars()(right_knot_index),
                this->brick_v_z_vars()(right_knot_index),
                this->brick_omega_x_vars()(right_knot_index),
                this->brick_v_y_vars()(left_knot_index),
                this->brick_v_z_vars()(left_knot_index),
                this->brick_omega_x_vars()(left_knot_index), f_FB_B_r,
                this->dt_(left_knot_index), &bound_vars);
            this->prog_->AddConstraint(constraint, bound_vars);
            break;
          }
          case IntegrationMethod::kMidpoint: {
            auto constraint =
                std::make_shared<BrickDynamicMidpointIntegrationConstraint>(
                    gripper_brick_, plant_mutable_contexts_[right_knot_index],
                    plant_mutable_contexts_[left_knot_index],
                    finger_face_contacts, finger_face_contacts,
                    brick_lid_friction_force_magnitude,
                    brick_lid_friction_torque_magnitude);
            VectorX<symbolic::Variable> bound_vars;
            constraint->ComposeX<symbolic::Variable>(
                this->q_vars_.col(right_knot_index),
                this->q_vars_.col(left_knot_index),
                this->brick_v_y_vars()(right_knot_index),
                this->brick_v_z_vars()(right_knot_index),
                this->brick_omega_x_vars()(right_knot_index),
                this->brick_v_y_vars()(left_knot_index),
                this->brick_v_z_vars()(left_knot_index),
                this->brick_omega_x_vars()(left_knot_index), f_FB_B_r, f_FB_B_l,
                this->dt_(left_knot_index), &bound_vars);
            this->prog_->AddConstraint(constraint, bound_vars);
            break;
          }
        }
      };
  // Now add the dynamic constraint
  int last_transition_end_knot = 0;
  for (const auto& finger_transition : sorted_finger_transitions) {
    for (int i = last_transition_end_knot;
         i < finger_transition->start_knot_index; ++i) {
      add_integration_constraint(
          i, finger_face_contacts_[last_transition_end_knot]);
    }
    // During the transition.
    for (int i = finger_transition->start_knot_index;
         i < finger_transition->end_knot_index; ++i) {
      // The active contacts are
      // finger_face_contacts_[finger_transition->start_knot_index + 1].
      DRAKE_DEMAND(finger_transition->end_knot_index >=
                   finger_transition->start_knot_index + 2);
      add_integration_constraint(
          i, finger_face_contacts_[finger_transition->start_knot_index + 1]);
    }
    last_transition_end_knot = finger_transition->end_knot_index;
  }
  // After the final transition, to nT.
  for (int i = last_transition_end_knot; i < nT_ - 1; ++i) {
    add_integration_constraint(i,
                               finger_face_contacts_[last_transition_end_knot]);
  }
}

void GripperBrickTrajectoryOptimization::
    AddBrickPositionIntegrationConstraint() {
  // First add the integration constraint on the brick position. We choose
  // midpoint integration.
  Vector3<symbolic::Variable> q_brick_l, v_brick_l, q_brick_r, v_brick_r;
  q_brick_l << q_vars_(gripper_brick_->brick_translate_y_position_index(), 0),
      q_vars_(gripper_brick_->brick_translate_z_position_index(), 0),
      q_vars_(gripper_brick_->brick_revolute_x_position_index(), 0);
  v_brick_l << brick_v_y_vars()(0), brick_v_z_vars()(0),
      brick_omega_x_vars()(0);
  auto position_midpoint_constraint = std::make_shared<
      systems::trajectory_optimization::MidPointIntegrationConstraint>(3);
  for (int i = 1; i < nT_; ++i) {
    q_brick_r << q_vars_(gripper_brick_->brick_translate_y_position_index(), i),
        q_vars_(gripper_brick_->brick_translate_z_position_index(), i),
        q_vars_(gripper_brick_->brick_revolute_x_position_index(), i);
    v_brick_r << brick_v_y_vars()(i), brick_v_z_vars()(i),
        brick_omega_x_vars()(i);
    VectorX<symbolic::Variable> constraint_x;
    position_midpoint_constraint->ComposeX<symbolic::Variable>(
        q_brick_r, q_brick_l, v_brick_r, v_brick_l, dt_(i - 1), &constraint_x);
    prog_->AddConstraint(position_midpoint_constraint, constraint_x);
    q_brick_l = q_brick_r;
    v_brick_l = v_brick_r;
  }
}

namespace {
void UpdatePositionBoundsOutsideFace(const BrickFace face,
                                     const Eigen::Vector3d& brick_size,
                                     double margin, Eigen::Vector3d* lower,
                                     Eigen::Vector3d* upper) {
  switch (face) {
    case BrickFace::kNegY: {
      (*upper)(1) = -brick_size(1) / 2 - margin;
      break;
    }
    case BrickFace::kNegZ: {
      (*upper)(2) = -brick_size(2) / 2 - margin;
      break;
    }
    case BrickFace::kPosY: {
      (*lower)(1) = brick_size(1) / 2 + margin;
      break;
    }
    case BrickFace::kPosZ: {
      (*lower)(2) = brick_size(2) / 2 + margin;
      break;
    }
  }
}
}  // namespace

void GripperBrickTrajectoryOptimization::AddKinematicInContactConstraint(
    double face_shrink_factor, double rolling_angle_bound, double depth) {
  // Add the initial contact constraint
  for (const auto& finger_face_contact : finger_face_contacts_[0]) {
    AddFingerTipInContactWithBrickFaceConstraint(
        *gripper_brick_, finger_face_contact.first, finger_face_contact.second,
        prog_.get(), q_vars_.col(0), plant_mutable_contexts_[0],
        face_shrink_factor, depth);
  }
  for (int i = 1; i < nT_; ++i) {
    for (const auto& finger_face_contact : finger_face_contacts_[i]) {
      // Check if this finger_face_contact is active at time i - 1. If so, then
      // we only allow rolling or sticking contact for this (finger, face) pair.
      auto it = finger_face_contacts_[i - 1].find(finger_face_contact.first);
      if (it == finger_face_contacts_[i - 1].end()) {
        // This finger just landed on the face at knot i.
        AddFingerTipInContactWithBrickFaceConstraint(
            *gripper_brick_, finger_face_contact.first,
            finger_face_contact.second, prog_.get(), q_vars_.col(i),
            plant_mutable_contexts_[i], face_shrink_factor);
      } else if (it->second == finger_face_contact.second) {
        // This finger sticks to the face.
        AddFingerNoSlidingConstraint(
            *gripper_brick_, finger_face_contact.first,
            finger_face_contact.second, rolling_angle_bound, prog_.get(),
            plant_mutable_contexts_[i - 1], plant_mutable_contexts_[i],
            q_vars_.col(i - 1), q_vars_.col(i), face_shrink_factor);
        // The middle-point also sticks to the face.
        AddFingerNoSlidingConstraint(
            *gripper_brick_, finger_face_contact.first,
            finger_face_contact.second, rolling_angle_bound, prog_.get(),
            plant_mutable_contexts_[i - 1], plant_mutable_contexts_mid_[i - 1],
            q_vars_.col(i - 1), q_mid_vars_.col(i - 1), face_shrink_factor);
      }
      // If the finger tip is in contact, we want to avoid that the finger link
      // 2 body is also in contact with the brick. We do this by requiring that
      // the origin of link2 is outside of the contact face.
      Eigen::Vector3d p_BLink2_lower(-kInf, -kInf, -kInf);
      Eigen::Vector3d p_BLink2_upper(kInf, kInf, kInf);
      UpdatePositionBoundsOutsideFace(finger_face_contact.second,
                                      gripper_brick_->brick_size(),
                                      gripper_brick_->finger_tip_radius(),
                                      &p_BLink2_lower, &p_BLink2_upper);
      prog_->AddConstraint(
          std::make_shared<multibody::PositionConstraint>(
              &(gripper_brick_->plant()), gripper_brick_->brick_frame(),
              p_BLink2_lower, p_BLink2_upper,
              gripper_brick_->finger_link2_frame(finger_face_contact.first),
              Eigen::Vector3d::Zero(), plant_mutable_contexts_[i]),
          q_vars_.col(i));
    }
  }
}

void GripperBrickTrajectoryOptimization::AddCollisionAvoidanceConstraint(
    const std::vector<const FingerTransition*>& sorted_finger_transitions,
    double collision_avoidance_margin) {
  for (const auto& finger_transition : sorted_finger_transitions) {
    SortedPair<geometry::GeometryId> geometry_pair(
        gripper_brick_->finger_tip_sphere_geometry_id(
            finger_transition->finger),
        gripper_brick_->brick_geometry_id());
    for (int i = finger_transition->start_knot_index + 1;
         i < finger_transition->end_knot_index; ++i) {
      // At the i'th knot, the finger is not in collision.
      prog_->AddConstraint(
          std::make_shared<multibody::DistanceConstraint>(
              &(gripper_brick_->plant()), geometry_pair,
              plant_mutable_contexts_[i], collision_avoidance_margin, kInf),
          q_vars_.col(i));
    }

    // The mid point posture (q[n] + q[n+1])/2, is also not in collision.
    for (int i = finger_transition->start_knot_index;
         i < finger_transition->end_knot_index; ++i) {
      AddCollisionAvoidanceForInterpolatedPosture(i, 0.5, geometry_pair,
                                                  collision_avoidance_margin);
    }
    // when the finger lands, if there is another finger on the same face, these
    // two fingers should be separated apart to avoid collision.
    const int landing_knot = finger_transition->end_knot_index;
    for (const auto& finger_face : finger_face_contacts_[landing_knot]) {
      if (finger_face.first != finger_transition->finger &&
          finger_face.second ==
              finger_face_contacts_[landing_knot][finger_transition->finger]) {
        SortedPair<geometry::GeometryId> finger_pair(
            gripper_brick_->finger_tip_sphere_geometry_id(
                finger_transition->finger),
            gripper_brick_->brick_geometry_id());
        prog_->AddConstraint(
            std::make_shared<multibody::DistanceConstraint>(
                &(gripper_brick_->plant()), finger_pair,
                plant_mutable_contexts_[landing_knot], 0.02, kInf),
            q_vars_.col(landing_knot));
      }
    }
    // For the knot just prior to landing, the finger should be "outside" of
    // both the taking-off face, and the landing face, so as to avoid cutting
    // the corner. For example, if the finger takes off from +Y face, and lands
    // on -Z face, then the position of the finger tip just before landing
    // should have positive y component, and negative z component in the brick
    // frame.
    Eigen::Vector3d p_BTip_lower(-kInf, -kInf, -kInf);
    Eigen::Vector3d p_BTip_upper(kInf, kInf, kInf);
    UpdatePositionBoundsOutsideFace(
        finger_face_contacts_[finger_transition->start_knot_index]
                             [finger_transition->finger],
        gripper_brick_->brick_size(),
        0.01 + gripper_brick_->finger_tip_radius(), &p_BTip_lower,
        &p_BTip_upper);
    UpdatePositionBoundsOutsideFace(finger_transition->to_face,
                                    gripper_brick_->brick_size(),
                                    0.01 + gripper_brick_->finger_tip_radius(),
                                    &p_BTip_lower, &p_BTip_upper);
    prog_->AddConstraint(
        std::make_shared<multibody::PositionConstraint>(
            &(gripper_brick_->plant()), gripper_brick_->brick_frame(),
            p_BTip_lower, p_BTip_upper,
            gripper_brick_->finger_link2_frame(finger_transition->finger),
            gripper_brick_->p_L2Fingertip(),
            plant_mutable_contexts_[finger_transition->end_knot_index - 1]),
        q_vars_.col(finger_transition->end_knot_index - 1));
  }
}

void GripperBrickTrajectoryOptimization::AddPositionDifferenceBound(
    int left_knot, int position_index, double bound) {
  DRAKE_DEMAND(bound >= 0);
  prog_->AddLinearConstraint(q_vars_(position_index, left_knot + 1) -
                                 q_vars_(position_index, left_knot),
                             -bound, bound);
}

void GripperBrickTrajectoryOptimization::AddFrictionConeConstraints(
    double friction_cone_shrink_factor) {
  for (int i = 0; i < nT_; ++i) {
    for (const auto& finger_face : finger_face_contacts_[i]) {
      AddFrictionConeConstraint(*gripper_brick_, finger_face.first,
                                finger_face.second,
                                f_FB_B_[i][finger_face.first],
                                friction_cone_shrink_factor, prog_.get());
    }
  }
}

void GripperBrickTrajectoryOptimization::
    AddCollisionAvoidanceForInterpolatedPosture(
        int left_knot, double fraction,
        const SortedPair<geometry::GeometryId>& geometry_pair,
        double minimal_distance) {
  const int nq = gripper_brick_->plant().num_positions();
  auto q_interpolated = prog_->NewContinuousVariables(nq);
  prog_->AddLinearEqualityConstraint((1 - fraction) * q_vars_.col(left_knot) +
                                         fraction * q_vars_.col(left_knot + 1) -
                                         q_interpolated,
                                     Eigen::VectorXd::Zero(nq));
  diagram_contexts_interpolated_.push_back(
      gripper_brick_->diagram().CreateDefaultContext());
  systems::Context<double>* plant_context_interpolated =
      &(gripper_brick_->diagram().GetMutableSubsystemContext(
          gripper_brick_->plant(),
          diagram_contexts_interpolated_.back().get()));
  prog_->AddConstraint(std::make_shared<multibody::DistanceConstraint>(
                           &(gripper_brick_->plant()), geometry_pair,
                           plant_context_interpolated, minimal_distance, kInf),
                       q_interpolated);
}

void GripperBrickTrajectoryOptimization::AddBrickStaticEquilibriumConstraint(
    int knot) {
  std::vector<std::pair<Finger, BrickFace>> finger_face_contacts_knot;
  finger_face_contacts_knot.reserve(finger_face_contacts_[knot].size());
  for (const auto& finger_face : finger_face_contacts_[knot]) {
    finger_face_contacts_knot.emplace_back(finger_face.first,
                                           finger_face.second);
  }
  auto constraint = std::make_shared<BrickStaticEquilibriumNonlinearConstraint>(
      *gripper_brick_, finger_face_contacts_knot,
      plant_mutable_contexts_[knot]);
  const int nq = gripper_brick_->plant().num_positions();
  VectorX<symbolic::Variable> bound_vars(nq +
                                         finger_face_contacts_knot.size() * 2);
  bound_vars.head(nq) = q_vars_.col(knot);
  for (int i = 0; i < static_cast<int>(finger_face_contacts_knot.size()); ++i) {
    bound_vars.segment<2>(nq + 2 * i) =
        f_FB_B_[knot].at(finger_face_contacts_knot[i].first);
  }
  prog_->AddConstraint(constraint, bound_vars);
}

Eigen::VectorXd GripperBrickTrajectoryOptimization::ReconstructTimeSolution(
    const solvers::MathematicalProgramResult& result) const {
  Eigen::VectorXd t_sol(nT_);
  t_sol(0) = 0;
  const Eigen::VectorXd dt_sol = result.GetSolution(dt_);
  for (int i = 0; i < nT_ - 1; ++i) {
    t_sol(i + 1) = t_sol(i) + dt_sol(i);
  }
  return t_sol;
}

trajectories::PiecewisePolynomial<double>
GripperBrickTrajectoryOptimization::ReconstructFingerTrajectory(
    const solvers::MathematicalProgramResult& result,
    const std::map<std::string, int>& finger_joint_name_to_row_index_map)
    const {
  const Eigen::MatrixXd q_sol = result.GetSolution(q_vars_);
  Matrix6X<double> finger_keyframes(6, nT_);
  finger_keyframes.row(
      finger_joint_name_to_row_index_map.at("finger1_BaseJoint")) =
      q_sol.row(gripper_brick_->finger_base_position_index(Finger::kFinger1));
  finger_keyframes.row(
      finger_joint_name_to_row_index_map.at("finger2_BaseJoint")) =
      q_sol.row(gripper_brick_->finger_base_position_index(Finger::kFinger2));
  finger_keyframes.row(
      finger_joint_name_to_row_index_map.at("finger3_BaseJoint")) =
      q_sol.row(gripper_brick_->finger_base_position_index(Finger::kFinger3));
  finger_keyframes.row(
      finger_joint_name_to_row_index_map.at("finger1_MidJoint")) =
      q_sol.row(gripper_brick_->finger_mid_position_index(Finger::kFinger1));
  finger_keyframes.row(
      finger_joint_name_to_row_index_map.at("finger2_MidJoint")) =
      q_sol.row(gripper_brick_->finger_mid_position_index(Finger::kFinger2));
  finger_keyframes.row(
      finger_joint_name_to_row_index_map.at("finger3_MidJoint")) =
      q_sol.row(gripper_brick_->finger_mid_position_index(Finger::kFinger3));

  const Eigen::VectorXd t_sol = ReconstructTimeSolution(result);
  return trajectories::PiecewisePolynomial<double>::FirstOrderHold(
      t_sol, finger_keyframes);
}

class QuadraitcPolynomialMidpointInterpolationConstraint
    : public solvers::Constraint {
 public:
  QuadraitcPolynomialMidpointInterpolationConstraint(int dim)
      : solvers::Constraint(dim, 5 * dim + 1, Eigen::VectorXd::Zero(dim),
                            Eigen::VectorXd::Zero(dim)),
        dim_{dim} {}

  ~QuadraitcPolynomialMidpointInterpolationConstraint() override {}

  template <typename T>
  void ComposeX(const Eigen::Ref<const VectorX<T>>& q_l,
                const Eigen::Ref<const VectorX<T>>& q_r,
                const Eigen::Ref<const VectorX<T>>& v_l,
                const Eigen::Ref<const VectorX<T>>& v_r,
                const Eigen::Ref<const VectorX<T>>& q_m, const T& dt,
                VectorX<T>* x) const {
    x->resize(num_vars());
    (*x) << q_l, q_r, v_l, v_r, q_m, dt;
  }

 private:
  template <typename T>
  void DoEvalGeneric(const Eigen::Ref<const VectorX<T>>& x,
                     VectorX<T>* y) const {
    const auto& q_l = x.head(dim_);
    const auto& q_r = x.segment(dim_, dim_);
    const auto& v_l = x.segment(2 * dim_, dim_);
    const auto& v_r = x.segment(3 * dim_, dim_);
    const auto& q_m = x.segment(4 * dim_, dim_);
    const auto& dt = x(5 * dim_);
    *y = q_m - (q_l + q_r) / 2 - (v_l - v_r) / 4 * dt;
  }

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override {
    DoEvalGeneric<double>(x, y);
  }

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override {
    DoEvalGeneric<AutoDiffXd>(x, y);
  }

  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
              VectorX<symbolic::Expression>* y) const override {
    DoEvalGeneric<symbolic::Expression>(x.cast<symbolic::Expression>(), y);
  }
  int dim_;
};

void GripperBrickTrajectoryOptimization::AddMiddlePointIntegrationConstraint() {
  // The finger position is interpolated as piecewise linear polynomial.
  for (int i = 0; i < nT_ - 1; ++i) {
    for (Finger finger :
         {Finger::kFinger1, Finger::kFinger2, Finger::kFinger3}) {
      const int base_index = gripper_brick_->finger_base_position_index(finger);
      prog_->AddLinearEqualityConstraint(
          Eigen::RowVector3d(1, 1, -2), 0,
          Vector3<symbolic::Variable>(q_vars_(base_index, i),
                                      q_vars_(base_index, i + 1),
                                      q_mid_vars_(base_index, i)));
      const int mid_index = gripper_brick_->finger_mid_position_index(finger);
      prog_->AddLinearEqualityConstraint(
          Eigen::RowVector3d(1, 1, -2), 0,
          Vector3<symbolic::Variable>(q_vars_(mid_index, i),
                                      q_vars_(mid_index, i + 1),
                                      q_mid_vars_(mid_index, i)));
    }
  }

  // The brick trajectory is interpolated as piecewise quadratic polynomial.
  for (int i = 0; i < nT_ - 1; ++i) {
    Vector3<symbolic::Variable> q_brick_l, q_brick_r, v_brick_l, v_brick_r,
        q_brick_m;
    q_brick_l << q_vars_(gripper_brick_->brick_translate_y_position_index(), i),
        q_vars_(gripper_brick_->brick_translate_z_position_index(), i),
        q_vars_(gripper_brick_->brick_revolute_x_position_index(), i);
    q_brick_r << q_vars_(gripper_brick_->brick_translate_y_position_index(),
                         i + 1),
        q_vars_(gripper_brick_->brick_translate_z_position_index(), i + 1),
        q_vars_(gripper_brick_->brick_revolute_x_position_index(), i + 1);
    v_brick_l << brick_v_y_vars()(i), brick_v_z_vars()(i),
        brick_omega_x_vars()(i);
    v_brick_r << brick_v_y_vars()(i + 1), brick_v_z_vars()(i + 1),
        brick_omega_x_vars()(i + 1);
    q_brick_m << q_mid_vars_(gripper_brick_->brick_translate_y_position_index(),
                             i),
        q_mid_vars_(gripper_brick_->brick_translate_z_position_index(), i),
        q_mid_vars_(gripper_brick_->brick_revolute_x_position_index(), i);
    VectorX<symbolic::Variable> x_vars;
    auto constraint =
        std::make_shared<QuadraitcPolynomialMidpointInterpolationConstraint>(3);
    constraint->ComposeX<symbolic::Variable>(
        q_brick_l, q_brick_r, v_brick_l, v_brick_r, q_brick_m, dt_(i), &x_vars);
    prog_->AddConstraint(constraint, x_vars);
  }
}

trajectories::PiecewisePolynomial<double>
GripperBrickTrajectoryOptimization::ReconstructFingerForceTrajectory(
    Finger finger, const solvers::MathematicalProgramResult& result) const {
  Eigen::Matrix2Xd f_Cf_B_sol(2, nT_);
  f_Cf_B_sol.setZero();
  for (int i = 0; i < nT_; ++i) {
    auto it = f_FB_B_[i].find(finger);
    if (it != f_FB_B_[i].end()) {
      f_Cf_B_sol.col(i) = -result.GetSolution(it->second);
    }
  }
  const Eigen::VectorXd t_sol = ReconstructTimeSolution(result);
  return trajectories::PiecewisePolynomial<double>::FirstOrderHold(t_sol,
                                                                   f_Cf_B_sol);
}

optional<BrickFace> GripperBrickTrajectoryOptimization::GetFingerContactFace(
    double t, Finger finger,
    const Eigen::Ref<const Eigen::VectorXd>& t_sol) const {
  for (int i = 0; i < nT_ - 1; ++i) {
    if (t > t_sol(i) && t < t_sol(i + 1)) {
      // Found the interval.
      auto it_l = finger_face_contacts_[i].find(finger);
      if (it_l == finger_face_contacts_[i].end()) {
        return {};
      }
      auto it_r = finger_face_contacts_[i + 1].find(finger);
      if (it_r == finger_face_contacts_[i].end()) {
        return {};
      }
      return it_l->second;
    }
  }
  for (int i = 0; i < nT_; ++i) {
    if (t == t_sol(i)) {
      auto it = finger_face_contacts_[i].find(finger);
      if (it == finger_face_contacts_[i].end()) {
        return {};
      } else {
        return it->second;
      }
    }
  }
  return {};
}

optional<std::pair<double, BrickFace>>
GripperBrickTrajectoryOptimization::GetFingerContactLocationOnBrickFace(
    double t, Finger finger, const Eigen::Ref<const Eigen::VectorXd>& t_sol,
    const Eigen::Ref<const Eigen::MatrixXd>& q_sol,
    systems::Context<double>* plant_mutable_context) const {
  // First find out the time interval.
  for (int i = 0; i < nT_ - 1; ++i) {
    if (t >= t_sol[i] && t <= t_sol[i + 1]) {
      BrickFace face;
      if (t == t_sol[i] || t == t_sol[i + 1]) {
        int index = t == t_sol[i] ? i : i + 1;
        auto it = finger_face_contacts_[index].find(finger);
        if (it == finger_face_contacts_[index].end()) {
          // Finger not in contact.
          return {};
        }
        face = it->second;
      } else {
        // The finger contact must be active at both the left and the right
        // knots.
        auto it_l = finger_face_contacts_[i].find(finger);
        if (it_l == finger_face_contacts_[i].end()) {
          return {};
        }
        auto it_r = finger_face_contacts_[i + 1].find(finger);
        if (it_r == finger_face_contacts_[i + 1].end()) {
          return {};
        }
        face = it_l->second;
      }

      const double fraction = (t - t_sol[i]) / (t_sol[i + 1] - t_sol[i]);
      const Eigen::VectorXd q =
          fraction * q_sol.col(i) + (1 - fraction) * q_sol.col(i + 1);
      gripper_brick_->plant().SetPositions(plant_mutable_context, q);
      Eigen::Vector3d p_BFingertip;
      gripper_brick_->plant().CalcPointsPositions(
          *plant_mutable_context, gripper_brick_->finger_link2_frame(finger),
          gripper_brick_->p_L2Fingertip(), gripper_brick_->brick_frame(),
          &p_BFingertip);
      switch (face) {
        case BrickFace::kNegY:
        case BrickFace::kPosY: {
          return std::make_pair(p_BFingertip(2), face);
        }
        case BrickFace::kNegZ:
        case BrickFace::kPosZ: {
          return std::make_pair(p_BFingertip(1), face);
        }
        default: { throw std::runtime_error("Unknown face."); }
      }
    }
  }
  return {};
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake