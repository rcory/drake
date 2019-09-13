#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "drake/lcm/drake_lcm_interface.h"

namespace drake {
namespace examples {
namespace planar_gripper {

void PublishFramesToLcm(
    const std::string &channel_name,
    const std::unordered_map<std::string, Eigen::Isometry3d> &name_to_frame_map,
    drake::lcm::DrakeLcmInterface *lcm);

void PublishFramesToLcm(
    const std::string &channel_name,
    const std::vector<Eigen::Isometry3d> &frames,
    const std::vector<std::string> &frame_names,
    drake::lcm::DrakeLcmInterface *lcm);

}  // namespace drake
}  // namespace examples
}  // planar_gripper
