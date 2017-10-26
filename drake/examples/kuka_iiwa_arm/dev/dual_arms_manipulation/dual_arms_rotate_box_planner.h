#pragma once

#include <memory>

#include "drake/multibody/rigid_body_tree.h"
#include "drake/multibody/parsers/model_instance_id_table.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
std::unique_ptr<RigidBodyTreed> ConstructDualArmAndBox();

void VisualizePosture(RigidBodyTreed* tree, const Eigen::Ref<const Eigen::VectorXd>& q_kuka1, const Eigen::Ref<const Eigen::VectorXd>& q_kuka2, const Eigen::Ref<Eigen::Matrix<double, 7, 1>>& q_box);

class DualArmsRotateBoxPlanner : public solvers::MathematicalProgram {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DualArmsRotateBoxPlanner)

  DualArmsRotateBoxPlanner(RigidBodyTreed* tree, int nT);

  ~DualArmsRotateBoxPlanner() {};

  void AddDynamicsConstraint() { DoAddDynamicsConstraint();}

 protected:
  virtual void DoAddDynamicsConstraint() = 0;

  RigidBodyTreed* tree_;
  int nT_;
  solvers::VectorXDecisionVariable dt_;
  solvers::MatrixDecisionVariable<7, Eigen::Dynamic> q_kuka1_;
  solvers::MatrixDecisionVariable<7, Eigen::Dynamic> q_kuka2_;
  solvers::MatrixDecisionVariable<7, Eigen::Dynamic> q_box_;
  solvers::MatrixDecisionVariable<6, Eigen::Dynamic> v_box_;
};

enum class IntegrationType {
  kBackwardEuler,
  kMidPoint,
  kCubicHermite,
};

class QuaternionJointInterpolationConstraint : public solvers::Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QuaternionJointInterpolationConstraint)

  QuaternionJointInterpolationConstraint(IntegrationType integration_type);

  ~QuaternionJointInterpolationConstraint() override {}

 protected:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd& y) const override;

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd& y) const override;

 private:
  IntegrationType integration_type_;
};

class CentroidalDynamicsConstraint : public solvers::Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CentroidalDynamicsConstraint)

  CentroidalDynamicsConstraint(double m, const Eigen::Ref<const Eigen::Matrix3d>& I, IntegrationType integration_type) :
      solvers::Constraint(6, 39, Eigen::Matrix<double, 6, 1>::Zero(), Eigen::Matrix<double, 6, 1>::Zero()),
      m_(m),
      I_(I),
      integration_type_(integration_type){}

  ~CentroidalDynamicsConstraint() override {}

 protected:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd& y) const override;

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd& y) const override;

 private:
  Eigen::Matrix<double, 6, 1> vdot(const Eigen::Ref<const Eigen::Matrix<double, 7, 1>>& q, const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& v, const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& wrench) const;

  AutoDiffVecd<Eigen::Dynamic, 6> vdot(const Eigen::Ref<const AutoDiffVecd<Eigen::Dynamic, 7>>& q, const Eigen::Ref<const AutoDiffVecd<Eigen::Dynamic, 6>>& v, const Eigen::Ref<const AutoDiffVecd<Eigen::Dynamic, 6>>& wrench) const;

  double m_;
  Eigen::Matrix3d I_;
  IntegrationType integration_type_;
  static constexpr double g_ = 9.81;
};

class CentroidalDynamicsContactImplicitDualArmsPlanner : public DualArmsRotateBoxPlanner {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CentroidalDynamicsContactImplicitDualArmsPlanner)

  CentroidalDynamicsContactImplicitDualArmsPlanner(RigidBodyTreed* tree, int nT);

 protected:
  void DoAddDynamicsConstraint();
};
}
}
}