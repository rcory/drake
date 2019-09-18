#include "drake/examples/planar_gripper/planar_gripper_common.h"

#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/lcmt_viewer_draw.hpp"

namespace drake {
namespace examples {
namespace planar_gripper {

using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;
using drake::multibody::MultibodyPlant;
using Eigen::Vector3d;

template <typename T>
void WeldGripperFrames(MultibodyPlant<T>* plant) {
  // The finger base links are all welded a fixed distance from the world
  // origin, on the Y-Z plane.
  const double kOriginToBaseDistance = 0.19;

  // Before welding, all finger base links sit at the world origin with the
  // finger pointing along the -Z axis, with all joint angles being zero.

  // Weld the first finger. Finger base links are arranged equidistant along the
  // perimeter of a circle. The first finger is welded 60 degrees from the
  // +Z-axis. Frames F1, F2, F3 correspond to the base link finger frames.
  RigidTransformd X_WF1(RollPitchYawd(M_PI / 3, 0, 0), Vector3d::Zero());
  X_WF1 = X_WF1 * RigidTransformd(Vector3d(0, 0, kOriginToBaseDistance));
  const multibody::Frame<T>& finger1_base_frame =
      plant->GetFrameByName("finger1_base");
  plant->WeldFrames(plant->world_frame(), finger1_base_frame, X_WF1);

  // Weld the second finger. The second finger is 120 degrees from the first
  // finger (i.e., the arc from finger 1's base to finger 2's base is 120
  // degrees).
  const RigidTransformd X_WF2 =
      RigidTransformd(RollPitchYawd(2 * M_PI / 3, 0, 0), Vector3d::Zero()) *
      X_WF1;
  const multibody::Frame<T>& finger2_base_frame =
      plant->GetFrameByName("finger2_base");
  plant->WeldFrames(plant->world_frame(), finger2_base_frame, X_WF2);

  // Weld the 3rd finger. The third finger is 120 degrees from the second
  // finger.
  const RigidTransformd X_WF3 =
      RigidTransformd(RollPitchYawd(2 * M_PI / 3, 0, 0), Vector3d::Zero()) *
      X_WF2;
  const multibody::Frame<T>& finger3_base_frame =
      plant->GetFrameByName("finger3_base");
  plant->WeldFrames(plant->world_frame(), finger3_base_frame, X_WF3);
}

// Explicit instantiations.
template void WeldGripperFrames(MultibodyPlant<double>* plant);

/// Build a keyframe matrix for joints in joint_ordering by extracting the
/// appropriate columns from all_keyframes. The interpretation of columns in
/// joint_keyframes are ordered as in joint_ordering.
/// @pre There are as many strings in headers as there are columns in
/// all_keyframes.
/// @pre Every string in joint_ordering is expected to be unique.
/// @pre Every string in joint_ordering is expected to be found in headers.
MatrixX<double> MakeKeyframes(MatrixX<double> all_keyframes,
                              std::vector<std::string> joint_ordering,
                              std::vector<std::string> headers) {
  // First find the columns in the keyframe data for just the joints in
  // joint_ordering.
  std::map<std::string, int> joint_name_to_col_index_map;
  for (const auto& header_name : joint_ordering) {
    auto match_it = std::find(headers.begin(), headers.end(), header_name);
    DRAKE_DEMAND(match_it != headers.end());
    joint_name_to_col_index_map[header_name] = match_it - headers.begin();
  }
  // Now create the keyframe matrix.
  const int keyframe_count = all_keyframes.rows();
  const int kNumFingerJoints = joint_ordering.size();
  MatrixX<double> joint_keyframes(keyframe_count, kNumFingerJoints);
  for (int i = 0; i < kNumFingerJoints; ++i) {
    const std::string& joint_name = joint_ordering[i];
    const int all_keyframe_col_index = joint_name_to_col_index_map[joint_name];
    joint_keyframes.block(0, i, keyframe_count, 1) =
        all_keyframes.block(0, all_keyframe_col_index, keyframe_count, 1);
  }
  return joint_keyframes;
}

std::pair<MatrixX<double>, std::map<std::string, int>> ParseKeyframes(
    const std::string& name, EigenPtr<Vector3<double>> brick_initial_pose) {
  const std::string keyframe_path = FindResourceOrThrow(name);
  std::fstream file;
  file.open(keyframe_path, std::fstream::in);
  DRAKE_DEMAND(file.is_open());

  // Count the number of lines in the file.
  std::string line;
  int line_count = 0;
  while (!std::getline(file, line).eof()) {
    line_count++;
  }
  const int keyframe_count = line_count - 1;
  drake::log()->info("Found {} keyframes", keyframe_count);

  // Get the file headers.
  file.clear();
  file.seekg(0);
  std::getline(file, line);
  std::stringstream sstream(line);
  std::vector<std::string> headers;
  std::string token;
  while (sstream >> token) {
    headers.push_back(token);
  }

  // Make sure we read the correct number of headers.
  const int kNumHeaders = 9;
  if (headers.size() != kNumHeaders) {
    throw std::runtime_error(
        "Unexpected number of headers found in keyframe input file.");
  }

  // Extract all keyframes (finger and brick)
  MatrixX<double> all_keyframes(keyframe_count, headers.size());
  for (int i = 0; i < all_keyframes.rows(); ++i) {
    for (int j = 0; j < all_keyframes.cols(); ++j) {
      file >> all_keyframes(i, j);
    }
  }

  // Find the columns in the keyframe data for just the finger joints and
  // create the corresponding keyframe matrix.
  std::vector<std::string> finger_joint_ordering = {
      "finger1_BaseJoint", "finger2_BaseJoint", "finger3_BaseJoint",
      "finger1_MidJoint",  "finger2_MidJoint",  "finger3_MidJoint"};
  MatrixX<double> finger_joint_keyframes =
      MakeKeyframes(all_keyframes, finger_joint_ordering, headers);

  // Find the columns in the keyframe data for just the brick joints and create
  // the corresponding keyframe matrix. Note: Only the first keyframe is used to
  // set the brick's initial position. All other brick keyframe data is unused.
  std::vector<std::string> brick_joint_ordering = {"brick_translate_y_joint",
                                                   "brick_translate_z_joint",
                                                   "brick_revolute_x_joint"};
  std::map<std::string, int> brick_joint_name_to_col_index_map;
  MatrixX<double> brick_joint_keyframes =
      MakeKeyframes(all_keyframes, brick_joint_ordering, headers);

  // Set the brick's initial positions.
  (*brick_initial_pose)(0) = brick_joint_keyframes(0, 0);  // y-translate
  (*brick_initial_pose)(1) = brick_joint_keyframes(0, 1);  // z-translate
  (*brick_initial_pose)(2) = brick_joint_keyframes(0, 2);  // x-revolute

  finger_joint_keyframes.transposeInPlace();

  // Create the finger joint name to column index map.
  std::map<std::string, int> finger_joint_name_to_col_index_map;
  for (size_t i = 0; i < finger_joint_ordering.size(); i++) {
    finger_joint_name_to_col_index_map[finger_joint_ordering[i]] = i;
  }

  return std::make_pair(finger_joint_keyframes,
                        finger_joint_name_to_col_index_map);
}

/// Utility to publish frames to LCM.
void PublishFramesToLcm(
    const std::string& channel_name,
    const std::unordered_map<std::string, Eigen::Isometry3d>& name_to_frame_map,
    drake::lcm::DrakeLcmInterface* lcm) {
  std::vector<Eigen::Isometry3d> poses;
  std::vector<std::string> names;
  for (const auto& pair : name_to_frame_map) {
    poses.push_back(pair.second);
    names.push_back(pair.first);
  }
  PublishFramesToLcm(channel_name, poses, names, lcm);
}

void PublishFramesToLcm(const std::string& channel_name,
                        const std::vector<Eigen::Isometry3d>& poses,
                        const std::vector<std::string>& names,
                        drake::lcm::DrakeLcmInterface* dlcm) {
  DRAKE_DEMAND(poses.size() == names.size());
  lcmt_viewer_draw frame_msg{};
  frame_msg.timestamp = 0;
  int32_t vsize = poses.size();
  frame_msg.num_links = vsize;
  frame_msg.link_name.resize(vsize);
  frame_msg.robot_num.resize(vsize, 0);

  for (size_t i = 0; i < poses.size(); i++) {
    Eigen::Isometry3f pose = poses[i].cast<float>();
    // Create a frame publisher
    Eigen::Vector3f goal_pos = pose.translation();
    Eigen::Quaternion<float> goal_quat =
        Eigen::Quaternion<float>(pose.linear());
    frame_msg.link_name[i] = names[i];
    frame_msg.position.push_back({goal_pos(0), goal_pos(1), goal_pos(2)});
    frame_msg.quaternion.push_back(
        {goal_quat.w(), goal_quat.x(), goal_quat.y(), goal_quat.z()});
  }

  const int num_bytes = frame_msg.getEncodedSize();
  const size_t size_bytes = static_cast<size_t>(num_bytes);
  std::vector<uint8_t> bytes(size_bytes);
  frame_msg.encode(bytes.data(), 0, num_bytes);
  dlcm->Publish(
      "DRAKE_DRAW_FRAMES_" + channel_name, bytes.data(), num_bytes, {});
}

/// Publishes frames once.
void PublishInitialFrames(systems::Context<double>& context,
                          multibody::MultibodyPlant<double>& plant,
                          lcm::DrakeLcm &lcm) {
  std::vector<std::string> body_names;
  std::vector<Eigen::Isometry3d> poses;

  // list the body names.
  body_names.push_back("brick_base_link");

  for (size_t i = 0; i < body_names.size(); i++) {
    auto& body = plant.GetBodyByName(body_names[i]);
    math::RigidTransform<double> X_WB = plant.EvalBodyPoseInWorld(context, body);
    poses.push_back(X_WB.GetAsIsometry3());
  }

  PublishFramesToLcm("SIM", poses, body_names, &lcm);
}


/// Visualizes the spatial forces via Evan's spatial force visualization PR.
ExternalSpatialToSpatialViz::ExternalSpatialToSpatialViz(
    MultibodyPlant<double>& plant, multibody::ModelInstanceIndex instance,
    double force_scale_factor)
    : plant_(plant),
      instance_(instance),
      force_scale_factor_(force_scale_factor) {
  // Make context with default parameters.
  plant_context_ = plant.CreateDefaultContext();
  this->DeclareAbstractInputPort(
      Value<std::vector<multibody::ExternallyAppliedSpatialForce<double>>>());
  this->DeclareVectorInputPort("x",
                               systems::BasicVector<double>(2 /* state */));
  this->DeclareAbstractOutputPort(&ExternalSpatialToSpatialViz::CalcOutput);
}

void ExternalSpatialToSpatialViz::CalcOutput(
    const systems::Context<double>& context,
    std::vector<multibody::SpatialForceOutput<double>>*
    spatial_forces_viz_output) const {
  auto external_spatial_forces_vec =
      this->get_input_port(0)
          .Eval<std::vector<multibody::ExternallyAppliedSpatialForce<double>>>(
              context);
  auto state = this->EvalVectorInput(context, 1)->get_value();
  plant_.SetPositionsAndVelocities(plant_context_.get(), instance_, state);
  spatial_forces_viz_output->clear();
  for (size_t i = 0; i < external_spatial_forces_vec.size(); i++) {
    // convert contact point from brick frame to world frame
    auto ext_spatial_force = external_spatial_forces_vec[i];
    //     drake::log()->info("p: \n{}", ext_spatial_force.p_BoBq_B);
    //     drake::log()->info("f: \n{}",
    //     ext_spatial_force.F_Bq_W.translational());

    // Compute the spatial force in the world frame.
    auto& body = plant_.get_body(ext_spatial_force.body_index);
    auto& X_WB = plant_.EvalBodyPoseInWorld(*plant_context_, body);
    auto p_BoBq_W = X_WB * ext_spatial_force.p_BoBq_B;

    spatial_forces_viz_output->emplace_back(
        p_BoBq_W,
        X_WB.rotation() * ext_spatial_force.F_Bq_W * force_scale_factor_);
  }
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
