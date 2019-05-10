/// @file
///
/// This file set up an example about control the allegro hand based on
/// position. In the program, the hand firstly grasps on a mug, and then twsits
/// it repeatly. The program presently only runs on simulation, with the file
/// allegro_single_object_simulation.cc which creats the simulation environment
/// for the hand and object. This program reads from LCM about the state of the
/// hands, and process command the positions of the finger joints through LCM.
/// It also uses the velocity states of the fingers to decide whether the hand
/// has finished the current motion, either by reaching the target position or
/// get stuck by collisions.

#include <Eigen/Dense>
#include "lcm/lcm-cpp.hpp"
#include <unistd.h>

#include "drake/examples/allegro_hand/allegro_common.h"
#include "drake/examples/allegro_hand/allegro_lcm.h"
#include "drake/lcmt_allegro_command.hpp"
#include "drake/lcmt_allegro_status.hpp"

namespace drake {
namespace examples {
namespace allegro_hand {
namespace {

const char* const kLcmStatusChannel = "ALLEGRO_STATUS";
const char* const kLcmCommandChannel = "ALLEGRO_COMMAND";

class PositionCommander {
 public:
  PositionCommander() {
    lcm_.subscribe(kLcmStatusChannel, &PositionCommander::HandleStatus,
                   this);
  }

  void Run() {
    allegro_command_.num_joints = kAllegroNumJoints;
    allegro_command_.joint_position.resize(kAllegroNumJoints, 0.);
    allegro_command_.num_torques = 0;
    allegro_command_.joint_torque.resize(0);

    flag_moving = true;
    Eigen::VectorXd target_joint_position(kAllegroNumJoints);
    target_joint_position.setZero();
    MovetoPositionUntilStuck(target_joint_position);
    usleep(1e6);

    // scissors
    target_joint_position.segment<4>(12) << 1.0, 0.6331, 1.3509, 1.0;
    target_joint_position.segment<4>(0) << 0.0885, 0.4, 0.6, -0.704;
    target_joint_position.segment<4>(4) << 0.0312, 0.4, 0.6, 0;
    target_joint_position.segment<4>(8) << 0.1019, 1.2375, 1.1346, 1.10244;
    MovetoPositionUntilStuck(target_joint_position);
    usleep(1e6);

    // paper
    target_joint_position.segment<4>(12) << 0.5284, 0.3693, 0.8977, 0.4863;
    target_joint_position.segment<4>(0) << -0.1220, 0.4, 0.6, -0.769;
    target_joint_position.segment<4>(4) << 0.0312, 0.4, 0.6, 0;
    target_joint_position.segment<4>(8) << 0.1767, 0.4, 0.6, -0.0528;
    MovetoPositionUntilStuck(target_joint_position);
    usleep(1e6);

    // rock
    target_joint_position.segment<4>(12) << 0.6017, 0.2976, 0.9034, 0.7929;
    target_joint_position.segment<4>(0) << -0.1194, 1.2068, 1.0, 1.4042;
    target_joint_position.segment<4>(4) << -0.0093, 1.2481, 1.4073, 0.8163;
    target_joint_position.segment<4>(8) << 0.1116, 1.2712, 1.3881, 1.0122;
    MovetoPositionUntilStuck(target_joint_position);
    usleep(1e6);

    std::cout << "Hand is closed. \n";
    while (0 == lcm_.handleTimeout(10)) {
    }

    // Record the joint position q when the fingers are close and gripping the
    // object
    Eigen::VectorXd close_hand_joint_position = Eigen::Map<Eigen::VectorXd>(
        &(allegro_status_.joint_position_measured[0]), kAllegroNumJoints);

  }

 private:
  inline void PublishPositionCommand(
      const Eigen::VectorXd& target_joint_position) {
    Eigen::VectorXd::Map(&allegro_command_.joint_position[0],
                         kAllegroNumJoints) = target_joint_position;
    lcm_.publish(kLcmCommandChannel, &allegro_command_);
  }

  inline void MovetoPositionUntilStuck(
      const Eigen::VectorXd& target_joint_position) {
    PublishPositionCommand(target_joint_position);
    // A time delay at the intial moving stage so that the noisy data from the
    // hand motion is filtered.
    for (int i = 0; i < 60; i++) {
      while (0 == lcm_.handleTimeout(10) || allegro_status_.utime == -1) {
      }
    }
    // wait until the fingers are stuck, or stop moving.
    while (flag_moving) {
      while (0 == lcm_.handleTimeout(10) || allegro_status_.utime == -1) {
      }
    }
  }

  void HandleStatus(const ::lcm::ReceiveBuffer*, const std::string&,
                    const lcmt_allegro_status* status) {
    allegro_status_ = *status;
    hand_state_.Update(allegro_status_);
    flag_moving = !hand_state_.IsAllFingersStuck();
  }

  ::lcm::LCM lcm_;
  lcmt_allegro_status allegro_status_;
  lcmt_allegro_command allegro_command_;
  AllegroHandMotionState hand_state_;

  bool flag_moving = true;
};

int do_main() {
  PositionCommander runner;
  runner.Run();
  return 0;
}

}  // namespace
}  // namespace allegro_hand
}  // namespace examples
}  // namespace drake

int main() { return drake::examples::allegro_hand::do_main(); }
