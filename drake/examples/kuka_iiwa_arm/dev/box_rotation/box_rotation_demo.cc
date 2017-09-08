/**
 * @file This file implements the box rotation demo.
 */

#include <iostream>
#include <list>
#include <memory>
#include <fstream>
#include <gflags/gflags.h>

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


DEFINE_string(keyframes, "", "Name of keyframe file to load");
DEFINE_string(urdf, "", "Name of keyframe file to load");

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace box_rotation {
namespace {

const char* const kKeyFramePath = "drake/examples/kuka_iiwa_arm/dev/box_rotation/"
    "working_keyframes.txt";
const char* const kIiwaUrdf =
    "drake/manipulation/models/iiwa_description/urdf/"
        "dual_iiwa14.urdf";

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

  typedef std::function<void(
      const robotlocomotion::robot_plan_t*)> IiwaPublishCallback;

  // creates the publisher
  IiwaPublishCallback iiwa_callback =
      ([&](const robotlocomotion::robot_plan_t* plan) {
        lcm.publish("COMMITTED_ROBOT_PLAN", plan);
      });

//  const std::string iiwa_path = FindResourceOrThrow(
//      "drake/manipulation/models/iiwa_description/urdf/"
//          "dual_iiwa14_primitive_collision.urdf");

  const std::string iiwa_path =
      (!FLAGS_urdf.empty() ? FLAGS_urdf : FindResourceOrThrow(kIiwaUrdf));

  // create the RBT
  auto tree = std::make_unique<RigidBodyTree<double>>();

  parsers::urdf::AddModelInstanceFromUrdfFileToWorld(
      iiwa_path, multibody::joints::kFixed, tree.get());

  // create a reference to the RBT
  const RigidBodyTree<double>& iiwa = *(tree.get());

  const std::string framesFile = (!FLAGS_keyframes.empty() ? FLAGS_keyframes :
                               FindResourceOrThrow(kKeyFramePath));

  MatrixX<double> allKeyFrames = get_posture(framesFile);

  // extract left and righ arm keyframes
  MatrixX<double>  keyframes(12,14);
  keyframes.block<12,7>(0,0) = allKeyFrames.block<12,7>(0,1);
  keyframes.block<12,7>(0,7) = allKeyFrames.block<12,7>(0,8);
  keyframes.transposeInPlace();

  //std::cout<<"keyframes.rows = "<<keyframes.rows()<<std::endl;

  const int N = (int)allKeyFrames.rows();
  std::vector<double> times((ulong)N);
  for (int i = 0; i < N; ++i) {
    if (i == 0)
      times[i] = allKeyFrames(i, 0);
    else
      times[i] = times[i - 1] + allKeyFrames(i, 0);
  }

  //std::cout<<"robot pos = "<<iiwa.get_num_positions()<<std::endl;

  std::vector<int> info(times.size(), 1);
  robotlocomotion::robot_plan_t plan{};

  *(&plan) = EncodeKeyFrames(iiwa, times, info, keyframes);
  iiwa_callback(&plan);

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

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::kuka_iiwa_arm::box_rotation::RunBoxRotationDemo();
  return 0;
}
