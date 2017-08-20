#include "drake/solvers/gurobi_solver.h"
#include "drake/multibody/rigid_body_constraint.h"
#include "drake/multibody/rigid_body_ik.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/manipulation/util/simple_tree_visualizer.h"
#include "drake/common/find_resource.h"
#include "drake/multibody/global_inverse_kinematics.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_tree_construction.h"

using Eigen::Vector3d;
using Eigen::Isometry3d;

using drake::solvers::SolutionResult;

namespace drake {
namespace multibody {
namespace {

std::unique_ptr<RigidBodyTree<double>> ConstructKuka() {
  std::unique_ptr<RigidBodyTree<double>> rigid_body_tree =
      std::make_unique<RigidBodyTree<double>>();
  const std::string model_path = FindResourceOrThrow(
      "drake/manipulation/models/iiwa_description/urdf/"
          "iiwa14_polytope_collision.urdf");

  parsers::urdf::AddModelInstanceFromUrdfFile(
      model_path,
      drake::multibody::joints::kFixed,
      nullptr,
      rigid_body_tree.get());

  AddFlatTerrainToWorld(rigid_body_tree.get());
  return rigid_body_tree;
}
/*
template<int kNumSampleX, int kNumSampleY, int kNumSampleZ>
std::array<Eigen::Vector3d, kNumSampleX * kNumSampleY * kNumSampleZ> GenerateEEposition() {
  constexpr int kNumSample = kNumSampleX * kNumSampleY * kNumSampleZ;
  Eigen::Matrix<double, kNumSampleZ, 1> sample_z = Eigen::Matrix<double, kNumSampleZ, 1>::LinSpaced(kNumSampleZ, -0.1, 0.1);
  Eigen::Matrix<double, kNumSampleX, 1> sample_x = Eigen::Matrix<double, kNumSampleX, 1>::LinSpaced(kNumSampleX, -0.8, 0.8);
  Eigen::Matrix<double, kNumSampleX, 1> sample_y = Eigen::Matrix<double, kNumSampleY, 1>::LinSpaced(kNumSampleY, -0.8, 0.8);
  std::array<Eigen::Vector3d, kNumSample> samples;
  for (int i = 0; i < kNumSampleX; ++i) {
    for (int j = 0; j < kNumSampleY; ++j) {
      for (int k = 0; k < kNumSampleZ; ++k) {
        samples[i * kNumSampleY * kNumSampleZ + j * kNumSampleZ + k] = Eigen::Vector3d(sample_x(i), sample_y(j), sample_z(k));
      }
    }
  }
  return samples;
};*/

void SolveNonlinearIK(RigidBodyTreed* robot, int ee_idx,
                      const Eigen::Vector3d& ee_pos,
                      const Eigen::Quaterniond& ee_quat, int* info,
                      Eigen::VectorXd* q_sol) {
  WorldPositionConstraint pos_cnstr(robot, ee_idx, Eigen::Vector3d::Zero(),
                                    ee_pos, ee_pos);
  WorldQuatConstraint orient_cnstr(
      robot, ee_idx,
      Eigen::Vector4d(ee_quat.w(), ee_quat.x(), ee_quat.y(), ee_quat.z()), 0);

  q_sol->resize(7);
  Eigen::VectorXd q_guess = Eigen::Matrix<double, 7, 1>::Zero();
  Eigen::VectorXd q_nom = q_guess;
  IKoptions ikoptions(robot);
  std::vector<std::string> infeasible_constraint;
  std::array<RigidBodyConstraint*, 2> cnstr = {{&pos_cnstr, &orient_cnstr}};
  inverseKin(robot, q_guess, q_nom, 2, cnstr.data(), ikoptions, q_sol, info,
             &infeasible_constraint);
}

int DoMain() {
  // First generate the sample end effector position
  /*constexpr int kNumSampleZ = 5;
  constexpr int kNumSampleX = 41;
  constexpr int kNumSampleY = 41;
  const auto ee_pos_samples = GenerateEEposition<kNumSampleX, kNumSampleY, kNumSampleZ>();
*/
  const auto tree = ConstructKuka();
  int ee_idx = tree->FindBodyIndex("iiwa_link_ee");

  drake::lcm::DrakeLcm lcm;

  manipulation::SimpleTreeVisualizer visualizer(*tree.get(), &lcm);

  //Eigen::Matrix<double, 7, 1> q = Eigen::Matrix<double, 7, 1>::Zero();
  //visualizer.visualize(q);
//  KinematicsCache<double> cache = tree->CreateKinematicsCache();
//  cache.initialize(q);
//  tree->doKinematics(cache);
//  const auto pos_origin = tree->transformPoints(cache, Eigen::Vector3d(0, 0, 0), ee_idx, 0);
//  const auto pos_pt = tree->transformPoints(cache, Eigen::Vector3d(0.1, 0, 0), ee_idx, 0);
//
//  std::cout << pos_origin.transpose() << std::endl;
//  std::cout << pos_pt.transpose() << std::endl;

  Eigen::Matrix3d ee_orient_des;
  ee_orient_des << 0, 1, 0,
                   0, 0, -1,
                   -1,0, 0;
  const Eigen::Quaterniond ee_quat_des(ee_orient_des);

  Eigen::VectorXd q_sol;
  int info;

  SolveNonlinearIK(tree.get(), ee_idx, Eigen::Vector3d(0, 0.8, 0), ee_quat_des, &info, &q_sol);
  std::cout << info << std::endl;
  visualizer.visualize(q_sol);
  std::cout << q_sol.transpose() << std::endl;

  return 0;
}
}  // namespace
}  // namespace multibody
}  // namespace drake

int main() {
  return drake::multibody::DoMain();
}