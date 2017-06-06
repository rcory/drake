#include "drake/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/dual_arms_rotate_box_planner.h"

#include "drake/common/drake_path.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/parsers/sdf_parser.h"
#include "drake/multibody/joints/fixed_joint.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/rigid_body_plant/viewer_draw_translator.h"
#include "drake/multibody/rigid_body_plant/create_load_robot_message.h"
#include "drake/lcmtypes/drake/lcmt_viewer_load_robot.hpp"
#include "drake/math/quaternion.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
std::unique_ptr<RigidBodyTreed> ConstructDualArmAndBox() {
  std::unique_ptr<RigidBodyTree<double>> rigid_body_tree =
      std::make_unique<RigidBodyTree<double>>();
  const std::string model_path = drake::GetDrakePath() +
                                 "/manipulation/models/iiwa_description/urdf/"
                                 "dual_iiwa14_polytope_collision.urdf";

  parsers::urdf::AddModelInstanceFromUrdfFile(model_path,
                                              drake::multibody::joints::kFixed,
                                              nullptr, rigid_body_tree.get());

  const std::string box_path =
      drake::GetDrakePath() +
      "/examples/kuka_iiwa_arm/dev/dual_arms_manipulation/box.urdf";

  parsers::urdf::AddModelInstanceFromUrdfFile(
      box_path, drake::multibody::joints::kQuaternion, nullptr,
      rigid_body_tree.get());

  return rigid_body_tree;
}

void VisualizePosture(RigidBodyTreed* tree,
                      const Eigen::Ref<const Eigen::VectorXd>& q_kuka1,
                      const Eigen::Ref<const Eigen::VectorXd>& q_kuka2,
                      const Eigen::Ref<Eigen::Matrix<double, 7, 1>>& q_box) {
  lcm::DrakeLcm lcm;
  std::vector<uint8_t> message_bytes;

  lcmt_viewer_load_robot load_msg =
      multibody::CreateLoadRobotMessage<double>(*tree);

  const int length = load_msg.getEncodedSize();
  message_bytes.resize(length);
  load_msg.encode(message_bytes.data(), 0, length);
  lcm.Publish("DRAKE_VIEWER_LOAD_ROBOT", message_bytes.data(),
              message_bytes.size());

  systems::ViewerDrawTranslator posture_drawer(*tree);
  Eigen::VectorXd x(tree->get_num_positions() + tree->get_num_velocities());
  x.block<7, 1>(tree->FindBody("left_iiwa_link_0")->get_position_start_index(),
                0) = q_kuka1;
  x.block<7, 1>(tree->FindBody("right_iiwa_link_0")->get_position_start_index(),
                0) = q_kuka2;
  x.block<7, 1>(tree->FindBody("box")->get_position_start_index(), 0) = q_box;
  systems::BasicVector<double> q_draw(x);
  posture_drawer.Serialize(0, q_draw, &message_bytes);
  lcm.Publish("DRAKE_VIEWER_DRAW", message_bytes.data(), message_bytes.size());
}

DualArmsRotateBoxPlanner::DualArmsRotateBoxPlanner(RigidBodyTreed* tree, int nT)
    : tree_(tree),
      nT_(nT),
      dt_(NewContinuousVariables(nT_, "dt")),
      q_kuka1_(NewContinuousVariables(7, nT_, "q_kuka1")),
      q_kuka2_(NewContinuousVariables(7, nT_, "q_kuka2")),
      q_box_(NewContinuousVariables(7, nT_, "q_box")),
      v_box_(NewContinuousVariables(7, nT_, "v_box")) {
  AddBoundingBoxConstraint(Eigen::VectorXd::Zero(nT_ - 1), Eigen::VectorXd::Constant(std::numeric_limits<double>::infinity(), nT_ - 1), dt_);
  for (int i = 0; i < nT_; ++i) {
    AddBoundingBoxConstraint(tree_->joint_limit_min, tree_->joint_limit_max, {q_kuka1_.col(i), q_kuka2_.col(i), q_box_.col(i)});
  }
}

void CentroidalDynamicsConstraint::DoEval(
    const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd& y) const {
  Eigen::Matrix<double, 7, 1> q_l = x.head<7>();
  Eigen::Matrix<double, 6, 1> v_l = x.block<6, 1>(7, 0);
  Eigen::Matrix<double, 6, 1> wrench_l = x.block<6, 1>(13, 0);
  Eigen::Matrix<double, 7, 1> q_r = x.block<7, 1>(19, 0);
  Eigen::Matrix<double, 6, 1> v_r = x.block<6, 1>(26, 0);
  Eigen::Matrix<double, 6, 1> wrench_r = x.block<6, 1>(32, 0);
  double dt = x(38);
  auto vdot_l = vdot(q_l, v_l, wrench_l);
  auto vdot_r = vdot(q_r, v_r, wrench_r);
  switch (integration_type_) {
    case IntegrationType::kBackwardEuler: {
      y = v_r - v_l - vdot_r * dt;
      break;
    }
    case IntegrationType::kMidPoint: {
      y = v_r - v_l - (vdot_l + vdot_r) * dt;
      break;
    }
    case IntegrationType::kCubicHermite: {
      throw std::runtime_error("Not implemented yet");
      break;
    }
  }
}

void CentroidalDynamicsConstraint::DoEval(
    const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd& y) const {
  AutoDiffVecd<Eigen::Dynamic, 7> q_l = x.head<7>();
  AutoDiffVecd<Eigen::Dynamic, 6> v_l = x.block<6, 1>(7, 0);
  AutoDiffVecd<Eigen::Dynamic, 6> wrench_l = x.block<6, 1>(13, 0);
  AutoDiffVecd<Eigen::Dynamic, 7> q_r = x.block<7, 1>(19, 0);
  AutoDiffVecd<Eigen::Dynamic, 6> v_r = x.block<6, 1>(26, 0);
  AutoDiffVecd<Eigen::Dynamic, 6> wrench_r = x.block<6, 1>(32, 0);
  auto dt = x(38);
  auto vdot_l = vdot(q_l, v_l, wrench_l);
  auto vdot_r = vdot(q_r, v_r, wrench_r);
  switch (integration_type_) {
    case IntegrationType::kBackwardEuler: {
      y = v_r - v_l - vdot_r * dt;
      break;
    }
    case IntegrationType::kMidPoint: {
      y = v_r - v_l - (vdot_l + vdot_r) * dt;
      break;
    }
    case IntegrationType::kCubicHermite: {
      throw std::runtime_error("Not implemented yet");
      break;
    }
  }
}

Eigen::Matrix<double, 6, 1> CentroidalDynamicsConstraint::vdot(
    const Eigen::Ref<const Eigen::Matrix<double, 7, 1>>& q,
    const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& v,
    const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& wrench) const {
  Eigen::Matrix<double, 6, 1> ret;
  ret.head<3>() = wrench.head<3>() / m_;
  ret(2) -= g_;
  Eigen::Vector3d I_wdot =
      wrench.tail<3>() - v.tail<3>().cross(I_ * v.tail<3>());
  ret.tail<3>() = I_.llt().solve(I_wdot);
  return ret;
}

AutoDiffVecd<Eigen::Dynamic, 6> CentroidalDynamicsConstraint::vdot(
    const Eigen::Ref<const AutoDiffVecd<Eigen::Dynamic, 7>>& q,
    const Eigen::Ref<const AutoDiffVecd<Eigen::Dynamic, 6>>& v,
    const Eigen::Ref<const AutoDiffVecd<Eigen::Dynamic, 6>>& wrench) const {
  AutoDiffVecd<Eigen::Dynamic, 6> ret;
  Eigen::AutoDiffScalar<Eigen::Matrix<double, Eigen::Dynamic, 1>> m_autodiff;
  m_autodiff.value() = m_;
  m_autodiff.derivatives() = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(19);
  ret.head<3>() = wrench.head<3>() / m_autodiff;
  ret(2).value() -= g_;
  AutoDiffVecd<Eigen::Dynamic, 3> I_wdot =
      wrench.tail<3>() -
      v.tail<3>().cross(
          I_.cast<Eigen::AutoDiffScalar<Eigen::Matrix<double, Eigen::Dynamic, 1>>>() *
          v.tail<3>());
  ret.tail<3>() = I_.cast<Eigen::AutoDiffScalar<Eigen::Matrix<double, Eigen::Dynamic, 1>>>()
                      .llt()
                      .solve(I_wdot);
  return ret;
}

QuaternionJointInterpolationConstraint::QuaternionJointInterpolationConstraint(IntegrationType integration_type) :
    solvers::Constraint(7, 27, Eigen::Matrix<double, 7, 1>::Zero(), Eigen::Matrix<double, 7, 1>::Zero()),
    integration_type_(integration_type) {}

void QuaternionJointInterpolationConstraint::DoEval(const Eigen::Ref<const Eigen::VectorXd> &x,
                                                    Eigen::VectorXd &y) const {
  Eigen::Matrix<double, 7, 1> q_l = x.head<7>();
  Eigen::Matrix<double, 6, 1> v_l = x.block<6, 1>(7, 0);
  Eigen::Matrix<double, 7, 1> q_r = x.block<7, 1>(13, 0);
  Eigen::Matrix<double, 6, 1> v_r = x.block<6, 1>(20, 0);
  double dt = x(26);
  y.resize(6);
  switch (integration_type_) {
    case IntegrationType::kBackwardEuler: {
      y.head<3>() = q_r.head<3>() - q_l.head<3>() - v_r.head<3>() * dt;
      double theta = v_r.tail<3>().norm() * dt;
      Eigen::Vector3d axis = v_r.tail<3>() / v_r.tail<3>().norm();
      Eigen::Vector4d quat_rotate;
      quat_rotate << cos(theta / 2), sin(theta / 2) * axis;
      y.tail<4>() =
          q_r.tail<4>() - math::quatProduct(q_l.tail<4>(), quat_rotate);
      break;
    }
    case IntegrationType::kMidPoint : {
      y.head<3>() = q_r.head<3>() - q_l.head<3>()
          - (v_r.head<3>() + v_l.head<3>()) / 2 * dt;
      Eigen::Vector3d omega_average = (v_r.tail<3>() + v_l.tail<3>()) / 2;
      double theta = omega_average.norm() * dt;
      Eigen::Vector3d axis = omega_average / omega_average.norm();
      Eigen::Vector4d quat_rotate;
      quat_rotate << cos(theta / 2), sin(theta / 2) * axis;
      y.tail<4>() =
          q_r.tail<4>() - math::quatProduct(q_l.tail<4>(), quat_rotate);
      break;
    }
    case IntegrationType::kCubicHermite: {
      throw std::runtime_error("Not implemented yet.");
    }
  }
}

void QuaternionJointInterpolationConstraint::DoEval(const Eigen::Ref<const AutoDiffVecXd> &x,
                                                    AutoDiffVecXd &y) const {
  using std::cos;
  using std::sin;
  AutoDiffVecd<Eigen::Dynamic, 7> q_l = x.head<7>();
  AutoDiffVecd<Eigen::Dynamic, 6> v_l = x.block<6, 1>(7, 0);
  AutoDiffVecd<Eigen::Dynamic, 7> q_r = x.block<7, 1>(13, 0);
  AutoDiffVecd<Eigen::Dynamic, 6> v_r = x.block<6, 1>(20, 0);
  auto dt = x(26);
  y.resize(6);
  switch (integration_type_) {
    case IntegrationType::kBackwardEuler: {
      y.head<3>() = q_r.head<3>() - q_l.head<3>() - v_r.head<3>() * dt;
      Eigen::AutoDiffScalar<Eigen::VectorXd> theta = v_r.tail<4>().norm() * dt;
      AutoDiffVecd<Eigen::Dynamic, 4> quat_rotate;
      quat_rotate << cos(theta / 2), sin(theta / 2) * v_r.tail<4>() / v_r.tail<4>().norm();
      y.tail<4>() = q_r.tail<4>() - math::quatProduct(q_l.tail<4>(), quat_rotate);
      break;
    }
    case IntegrationType::kMidPoint: {
      y.head<3>() = q_r.head<3>() - q_l.head<3>() - (v_l.head<3>() + v_r.head<3>()) * dt;
      AutoDiffVecd<Eigen::Dynamic, 3> omega_average = (v_l.tail<3>() + v_r.tail<3>()) / 2;
      Eigen::AutoDiffScalar<Eigen::VectorXd> theta = omega_average.norm() * dt;
      AutoDiffVecd<Eigen::Dynamic, 4> quat_rotate;
      quat_rotate << cos(theta / 2), sin(theta / 2) * omega_average / omega_average.norm();
      y.tail<4>() = q_r.tail<4>() - math::quatProduct(q_l.tail<4>(), quat_rotate);
      break;
    }
    case IntegrationType::kCubicHermite: {
      throw std::runtime_error("Not implemented.");
    }
  }
}

CentroidalDynamicsContactImplicitDualArmsPlanner::
    CentroidalDynamicsContactImplicitDualArmsPlanner(RigidBodyTreed* tree,
                                                     int nT)
    : DualArmsRotateBoxPlanner(tree, nT) {}

void CentroidalDynamicsContactImplicitDualArmsPlanner::
    DoAddDynamicsConstraint() {

}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake