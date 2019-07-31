#include "drake/examples/planar_gripper/gripper_brick_trajectory_optimization.h"

#include <limits>
#include <set>
#include <utility>

#include "drake/examples/planar_gripper/brick_dynamic_constraint.h"
#include "drake/examples/planar_gripper/brick_static_equilibrium_constraint.h"
#include "drake/examples/planar_gripper/gripper_brick_planning_utils.h"
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
      finger_face_contacts_(nT_) {
  // Create the contexts used at each knot.
  diagram_contexts_.reserve(nT_);
  for (int i = 0; i < nT_; ++i) {
    diagram_contexts_.push_back(
        gripper_brick->diagram().CreateDefaultContext());
    plant_mutable_contexts_[i] =
        &(gripper_brick_->diagram().GetMutableSubsystemContext(
            gripper_brick_->plant(), diagram_contexts_[i].get()));
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
                                  options.rolling_angle_bound);

  // During finger transition, the transition finger should avoid the brick.
  AddCollisionAvoidanceConstraint(sorted_finger_transitions,
                                  options.collision_avoidance_margin);

  // Add friction cone constraint.
  AddFrictionConeConstraints();
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
                this->brick_v_y_vars_(right_knot_index),
                this->brick_v_z_vars_(right_knot_index),
                this->brick_omega_x_vars_(right_knot_index),
                this->brick_v_y_vars_(left_knot_index),
                this->brick_v_z_vars_(left_knot_index),
                this->brick_omega_x_vars_(left_knot_index), f_FB_B_r,
                this->dt_(left_knot_index), &bound_vars);
            this->prog_->AddConstraint(constraint, bound_vars);
            break;
          }
          case IntegrationMethod::kMidpoint: {
            throw std::runtime_error(
                "midpoint integration is not supported yet");
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
  v_brick_l << brick_v_y_vars_(0), brick_v_z_vars_(0), brick_omega_x_vars_(0);
  auto position_midpoint_constraint = std::make_shared<
      systems::trajectory_optimization::MidPointIntegrationConstraint>(3);
  for (int i = 1; i < nT_; ++i) {
    q_brick_r << q_vars_(gripper_brick_->brick_translate_y_position_index(), i),
        q_vars_(gripper_brick_->brick_translate_z_position_index(), i),
        q_vars_(gripper_brick_->brick_revolute_x_position_index(), i);
    v_brick_r << brick_v_y_vars_(i), brick_v_z_vars_(i), brick_omega_x_vars_(i);
    VectorX<symbolic::Variable> constraint_x;
    position_midpoint_constraint->ComposeX<symbolic::Variable>(
        q_brick_r, q_brick_l, v_brick_r, v_brick_l, dt_(i - 1), &constraint_x);
    prog_->AddConstraint(position_midpoint_constraint, constraint_x);
    q_brick_l = q_brick_r;
    v_brick_l = v_brick_r;
  }
}

void GripperBrickTrajectoryOptimization::AddKinematicInContactConstraint(
    double face_shrink_factor, double rolling_angle_bound) {
  // Add the initial contact constraint
  for (const auto& finger_face_contact : finger_face_contacts_[0]) {
    AddFingerTipInContactWithBrickFace(
        *gripper_brick_, finger_face_contact.first, finger_face_contact.second,
        prog_.get(), q_vars_.col(0), plant_mutable_contexts_[0],
        face_shrink_factor);
  }
  for (int i = 1; i < nT_; ++i) {
    for (const auto& finger_face_contact : finger_face_contacts_[i]) {
      // Check if this finger_face_contact is active at time i - 1. If so, then
      // we only allow rolling or sticking contact for this (finger, face) pair.
      auto it = finger_face_contacts_[i - 1].find(finger_face_contact.first);
      if (it == finger_face_contacts_[i - 1].end()) {
        // This finger just landed on the face at knot i.
        AddFingerTipInContactWithBrickFace(
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
      }
      // If the finger tip is in contact, we want to avoid that the finger link
      // 2 body is not in contact with the brick. We do this by requiring that
      // the origin of link2 is outside of the contact face.
      Eigen::Vector3d p_BLink2_lower, p_BLink2_upper;
      switch (finger_face_contact.second) {
        case BrickFace::kNegY: {
          p_BLink2_lower << -kInf, -kInf, -kInf;
          p_BLink2_upper << kInf, -gripper_brick_->brick_size()(1) / 2, kInf;
          break;
        }
        case BrickFace::kNegZ: {
          p_BLink2_lower << -kInf, -kInf, -kInf;
          p_BLink2_upper << kInf, kInf, -gripper_brick_->brick_size()(2) / 2;
          break;
        }
        case BrickFace::kPosY: {
          p_BLink2_lower << -kInf, gripper_brick_->brick_size()(1) / 2, -kInf;
          p_BLink2_upper << kInf, kInf, kInf;
          break;
        }
        case BrickFace::kPosZ: {
          p_BLink2_lower << -kInf, -kInf, gripper_brick_->brick_size()(2) / 2;
          p_BLink2_upper << kInf, kInf, kInf;
          break;
        }
      }
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
        gripper_brick_->plant().GetCollisionGeometriesForBody(
            gripper_brick_->brick_frame().body())[0],
        gripper_brick_->plant().GetCollisionGeometriesForBody(
            gripper_brick_->finger_link2_frame(finger_transition->finger)
                .body())[0]);
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
    const int nq = gripper_brick_->plant().num_positions();
    for (int i = finger_transition->start_knot_index;
         i < finger_transition->end_knot_index; ++i) {
      auto q_middle = prog_->NewContinuousVariables(nq);
      prog_->AddLinearEqualityConstraint(
          q_vars_.col(i) + q_vars_.col(i + 1) - 2.0 * q_middle,
          Eigen::VectorXd::Zero(nq));
      diagram_contexts_midpoint_.push_back(
          gripper_brick_->diagram().CreateDefaultContext());
      systems::Context<double>* plant_context_midpoint =
          &(gripper_brick_->diagram().GetMutableSubsystemContext(
              gripper_brick_->plant(),
              diagram_contexts_midpoint_.back().get()));
      prog_->AddConstraint(
          std::make_shared<multibody::DistanceConstraint>(
              &(gripper_brick_->plant()), geometry_pair, plant_context_midpoint,
              collision_avoidance_margin, kInf),
          q_middle);
    }
    // when the finger lands, if there is another finger on the same face, these
    // two fingers should be separated apart to avoid collision.
    const int landing_knot = finger_transition->end_knot_index;
    for (const auto& finger_face : finger_face_contacts_[landing_knot]) {
      if (finger_face.first != finger_transition->finger &&
          finger_face.second ==
              finger_face_contacts_[landing_knot][finger_transition->finger]) {
        SortedPair<geometry::GeometryId> finger_pair(
            gripper_brick_->plant().GetCollisionGeometriesForBody(
                gripper_brick_->finger_link2_frame(finger_face.first)
                    .body())[0],
            gripper_brick_->plant().GetCollisionGeometriesForBody(
                gripper_brick_->finger_link2_frame(finger_transition->finger)
                    .body())[0]);
        prog_->AddConstraint(
            std::make_shared<multibody::DistanceConstraint>(
                &(gripper_brick_->plant()), finger_pair,
                plant_mutable_contexts_[landing_knot], 0.01, kInf),
            q_vars_.col(landing_knot));
      }
    }
  }
}

void GripperBrickTrajectoryOptimization::AddPositionDifferenceBound(
    int left_knot, int position_index, double bound) {
  DRAKE_DEMAND(bound >= 0);
  prog_->AddLinearConstraint(q_vars_(position_index, left_knot + 1) -
                                 q_vars_(position_index, left_knot),
                             -bound, bound);
}

void GripperBrickTrajectoryOptimization::AddFrictionConeConstraints() {
  for (int i = 0; i < nT_; ++i) {
    for (const auto& finger_face : finger_face_contacts_[i]) {
      AddFrictionConeConstraint(*gripper_brick_, finger_face.first,
                                finger_face.second,
                                f_FB_B_[i][finger_face.first], prog_.get());
    }
  }
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
