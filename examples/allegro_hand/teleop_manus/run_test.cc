/// @file
///
/// This file set up an example about control the allegro hand based on
/// position.

#include <gflags/gflags.h>

#include <Eigen/Dense>
#include <unistd.h>
#include "lcm/lcm-cpp.hpp"

#include "drake/examples/allegro_hand/allegro_common.h"
#include "drake/examples/allegro_hand/allegro_lcm.h"
#include "drake/lcmt_allegro_command.hpp"
#include "drake/lcmt_allegro_status.hpp"

namespace drake {
namespace examples {
namespace allegro_hand {
namespace {

const char* const kLcmCommandChannel = "ALLEGRO_COMMAND";

DEFINE_int64(joint_index, 0,
        "Joint index to move");

class PositionCommander {
 public:
  PositionCommander() {

  }

  void Run() {
    allegro_command_.num_joints = kAllegroNumJoints;
    allegro_command_.joint_position.resize(kAllegroNumJoints, 0.);
    allegro_command_.num_torques = 0;
    allegro_command_.joint_torque.resize(0);

    Eigen::VectorXd target_joint_position(kAllegroNumJoints);
    target_joint_position.setZero();


    while (true) {
      target_joint_position(FLAGS_joint_index) = 1;
      PublishPositionCommand(target_joint_position);
      sleep(1);
      target_joint_position.setZero();
      PublishPositionCommand(target_joint_position);
      sleep(1);
    }
  }

 private:
  inline void PublishPositionCommand(
      const Eigen::VectorXd& target_joint_position) {
    Eigen::VectorXd::Map(&allegro_command_.joint_position[0],
                         kAllegroNumJoints) = target_joint_position;
    lcm_.publish(kLcmCommandChannel, &allegro_command_);
  }

  ::lcm::LCM lcm_;
  lcmt_allegro_command allegro_command_;

};

int DoMain() {
  PositionCommander runner;
  runner.Run();
  return 0;
}

}  // namespace
}  // namespace allegro_hand
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    drake::examples::allegro_hand::DoMain();
    return 0;
}
