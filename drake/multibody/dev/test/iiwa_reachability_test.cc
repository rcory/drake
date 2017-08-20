#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "drake/common/find_resource.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/manipulation/util/simple_tree_visualizer.h"
#include "drake/multibody/global_inverse_kinematics.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_constraint.h"
#include "drake/multibody/rigid_body_ik.h"
#include "drake/multibody/rigid_body_tree_construction.h"
#include "drake/solvers/gurobi_solver.h"

using Eigen::Vector3d;
using Eigen::Isometry3d;

using drake::solvers::SolutionResult;

namespace drake {
namespace multibody {
namespace {
void RemoveFileIfExist(const std::string& file_name) {
  std::ifstream file(file_name);
  if (file) {
    if (remove(file_name.c_str()) != 0) {
      throw std::runtime_error("Error deleting file " + file_name);
    }
  }
  file.close();
}

std::unique_ptr<RigidBodyTree<double>> ConstructKuka() {
  std::unique_ptr<RigidBodyTree<double>> rigid_body_tree =
      std::make_unique<RigidBodyTree<double>>();
  const std::string model_path = FindResourceOrThrow(
      "drake/manipulation/models/iiwa_description/urdf/"
      "iiwa14_polytope_collision.urdf");

  parsers::urdf::AddModelInstanceFromUrdfFile(model_path,
                                              drake::multibody::joints::kFixed,
                                              nullptr, rigid_body_tree.get());

  AddFlatTerrainToWorld(rigid_body_tree.get());
  return rigid_body_tree;
}

template <int kNumSampleX, int kNumSampleY, int kNumSampleZ>
std::array<Eigen::Vector3d, kNumSampleX * kNumSampleY * kNumSampleZ>
GenerateEEposition() {
  constexpr int kNumSample = kNumSampleX * kNumSampleY * kNumSampleZ;
  Eigen::Matrix<double, kNumSampleZ, 1> sample_z =
      Eigen::Matrix<double, kNumSampleZ, 1>::LinSpaced(kNumSampleZ, -0.1, 0.1);
  Eigen::Matrix<double, kNumSampleX, 1> sample_x =
      Eigen::Matrix<double, kNumSampleX, 1>::LinSpaced(kNumSampleX, -0.8, 0.8);
  Eigen::Matrix<double, kNumSampleX, 1> sample_y =
      Eigen::Matrix<double, kNumSampleY, 1>::LinSpaced(kNumSampleY, -0.8, 0.8);
  std::array<Eigen::Vector3d, kNumSample> samples;
  for (int i = 0; i < kNumSampleX; ++i) {
    for (int j = 0; j < kNumSampleY; ++j) {
      for (int k = 0; k < kNumSampleZ; ++k) {
        samples[i * kNumSampleY * kNumSampleZ + j * kNumSampleZ + k] =
            Eigen::Vector3d(sample_x(i), sample_y(j), sample_z(k));
      }
    }
  }
  return samples;
};

void SolveNonlinearIK(RigidBodyTreed* robot, int ee_idx,
                      const Eigen::Vector3d& ee_pos,
                      const Eigen::Quaterniond& ee_quat,
                      const Eigen::VectorXd& q_guess, int* info,
                      Eigen::VectorXd* q_sol) {
  WorldPositionConstraint pos_cnstr(robot, ee_idx, Eigen::Vector3d::Zero(),
                                    ee_pos, ee_pos);
  WorldQuatConstraint orient_cnstr(
      robot, ee_idx,
      Eigen::Vector4d(ee_quat.w(), ee_quat.x(), ee_quat.y(), ee_quat.z()), 0);

  q_sol->resize(7);
  Eigen::VectorXd q_nom = q_guess;
  IKoptions ikoptions(robot);
  std::vector<std::string> infeasible_constraint;
  std::array<RigidBodyConstraint*, 2> cnstr = {{&pos_cnstr, &orient_cnstr}};
  inverseKin(robot, q_guess, q_nom, 2, cnstr.data(), ikoptions, q_sol, info,
             &infeasible_constraint);
}

int DoMain() {
  // First generate the sample end effector position
  constexpr int kNumSampleZ = 5;
  constexpr int kNumSampleX = 41;
  constexpr int kNumSampleY = 41;
  const auto ee_pos_samples =
      GenerateEEposition<kNumSampleX, kNumSampleY, kNumSampleZ>();

  const auto tree = ConstructKuka();
  // int ee_idx = tree->FindBodyIndex("iiwa_link_ee");

  int link6_idx = tree->FindBodyIndex("iiwa_link_6");
  drake::lcm::DrakeLcm lcm;

  manipulation::SimpleTreeVisualizer visualizer(*tree.get(), &lcm);

  // on ee, the palm direction is body x axis
  // on link6, the palm direction is body y axis
  Eigen::Matrix3d ee_orient_des;
  ee_orient_des << 0, 1, 0, 0, 0, -1, -1, 0, 0;
  const Eigen::Quaterniond ee_quat_des(ee_orient_des);

  Eigen::Matrix3d link6_orient_des;
  link6_orient_des << 1, 0, 0, 0, 0, 1, 0, -1, 0;
  const Eigen::Quaterniond link6_quat_des(link6_orient_des);

  Eigen::VectorXd q0 = Eigen::VectorXd::Zero(7);
  Eigen::VectorXd q_sol;
  int nonlinear_ik_info;

  SolveNonlinearIK(tree.get(), link6_idx, Eigen::Vector3d(0, 0.5, 0.5),
                   link6_quat_des, q0, &nonlinear_ik_info, &q_sol);
  visualizer.visualize(q_sol);

  GlobalInverseKinematics global_ik(*tree);

  const std::string output_file_name{"iiwa_reachability_global_ik.txt"};
  RemoveFileIfExist(output_file_name);
  std::fstream output_file;
  output_file.open(output_file_name, std::ios::app | std::ios::out);

  auto pos_cnstr = global_ik.AddBoundingBoxConstraint(
      0, 0, global_ik.body_position(link6_idx));
  const auto& link6_R = global_ik.body_rotation_matrix(link6_idx);
  Eigen::Matrix<double, 9, 1> link6_rotmat_des_flat;
  link6_rotmat_des_flat << link6_orient_des.col(0), link6_orient_des.col(1),
      link6_orient_des.col(2);
  solvers::VectorDecisionVariable<9> link6_R_flat;
  link6_R_flat << link6_R.col(0), link6_R.col(1), link6_R.col(2);
  global_ik.AddBoundingBoxConstraint(link6_rotmat_des_flat,
                                     link6_rotmat_des_flat, link6_R_flat);
  int pos_sample_count = 0;
  for (const auto& pos_sample : ee_pos_samples) {
    SolveNonlinearIK(tree.get(), link6_idx, pos_sample, link6_quat_des, q0,
                     &nonlinear_ik_info, &q_sol);
    solvers::SolutionResult global_ik_result{
        solvers::SolutionResult::kSolutionFound};
    int nonlinear_ik_resolve_info = 0;
    if (nonlinear_ik_info > 10) {
      pos_cnstr.constraint()->UpdateLowerBound(pos_sample);
      pos_cnstr.constraint()->UpdateUpperBound(pos_sample);
      solvers::GurobiSolver gurobi_solver;
      global_ik_result = gurobi_solver.Solve(global_ik);
      if (global_ik_result == SolutionResult::kSolutionFound) {
        Eigen::VectorXd q_global_ik =
            global_ik.ReconstructGeneralizedPositionSolution();
        SolveNonlinearIK(tree.get(), link6_idx, pos_sample, link6_quat_des,
                         q_global_ik, &nonlinear_ik_resolve_info, &q_sol);
      }
    }
    if (output_file.is_open()) {
      output_file << "\nposition:\n" << pos_sample.transpose() << std::endl;
      output_file << "nonlinear ik info:" << nonlinear_ik_info << std::endl;
      output_file << "global_ik info:" << global_ik_result << std::endl;
      output_file << "nonlinear ik resolve info:"
                  << nonlinear_ik_resolve_info << std::endl;
      output_file << std::endl;
    }
    std::cout << "pos count: " << pos_sample_count << std::endl;
    pos_sample_count++;
  }

  output_file.close();
  return 0;
}
}  // namespace
}  // namespace multibody
}  // namespace drake

int main() { return drake::multibody::DoMain(); }