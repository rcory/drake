/**
 * @file This file implements a state machine that drives the kuka iiwa arm to
 * pick up a block from one table to place it on another repeatedly.
 */

#include <iostream>
#include <list>
#include <memory>
#include <fstream>

#include <lcm/lcm-cpp.hpp>
#include "bot_core/robot_state_t.hpp"

#include "drake/examples/kuka_iiwa_arm/iiwa_common.h"

#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/common/drake_assert.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/find_resource.h"
#include "drake/lcmt_iiwa_status.hpp"
#include "drake/util/lcmUtil.h"
#include "robotlocomotion/robot_plan_t.hpp"
#include "external/lcmtypes_robotlocomotion/lcmtypes/robotlocomotion/robot_plan_t.hpp"


namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace box_rotation {
namespace {

MatrixX<double> get_posture(const std::string& name) {
  std::fstream fs;
  fs.open(name, std::fstream::in);
  DRAKE_DEMAND(fs.is_open());

  MatrixX<double> ret(12, 21);
  for (int i = 0; i < ret.rows(); ++i) {
    for (int j = 0; j < ret.cols(); ++j) {
      fs >> ret(i, j);
    }
  }
  return ret;
}

// Makes a state machine that drives the iiwa to pick up a block from one table
// and place it on the other table.
void RunBoxRotationDemo() {
  lcm::LCM lcm;

  typedef std::function<void(const robotlocomotion::robot_plan_t*)> IiwaPublishCallback;

  // creates the publisher
  IiwaPublishCallback iiwa_callback =
      ([&](const robotlocomotion::robot_plan_t* plan) {
        lcm.publish("COMMITTED_ROBOT_PLAN", plan);
      });

  const std::string iiwa_path = FindResourceOrThrow(
      "drake/manipulation/models/iiwa_description/urdf/"
          "dual_iiwa14_polytope_collision.urdf");

  auto tree = std::make_unique<RigidBodyTree<double>>();
  parsers::urdf::AddModelInstanceFromUrdfFileToWorld(
      iiwa_path, multibody::joints::kFixed, tree.get());

  //const RigidBodyTree<double>& iiwa = *(tree.get());

  MatrixX<double> keyframes = get_posture(
      "drake/examples/kuka_iiwa_arm/dev/box_rotation/simple_keyframes.txt");

  const int N = keyframes.rows();
  std::vector<double> times(N);
  for (int i = 0; i < N; ++i) {
    if (i == 0)
      times[i] = keyframes(i, 0);
    else
      times[i] = times[i - 1] + keyframes(i, 0);
  }


//  std::vector<MatrixX<double>> l_knots(N, MatrixX<double>::Zero(7, 1));
//  for (int i = 0; i < N; ++i)
//    l_knots[i] = keyframes.block<1, 7>(i, 1).transpose();
//
//  std::vector<MatrixX<double>> r_knots(N, MatrixX<double>::Zero(7, 1));
//  for (int i = 0; i < N; ++i)
//    r_knots[i] = keyframes.block<1, 7>(i, 8).transpose();
//
//  std::vector<int> info(times.size(), 1);
//  robotlocomotion::robot_plan_t plan{};
//
//  *(&plan) = EncodeKeyFrames(iiwa, times, info, keyframes);
//
//  iiwa_callback(&plan);

  // lcm handle loop
  while (true) {
    // Handles all messages.
    while (lcm.handleTimeout(10) == 0) {}
    // do stuff here
  }
}

}  // namespace
}  // namespace box_rotation
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main() {
  drake::examples::kuka_iiwa_arm::box_rotation::RunBoxRotationDemo();
  return 0;
}
