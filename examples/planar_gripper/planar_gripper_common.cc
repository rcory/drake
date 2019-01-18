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

std::string to_string(Finger finger) {
  switch (finger) {
    case Finger::kFinger1: {
      return "finger1";
    }
    case Finger::kFinger2: {
      return "finger2";
    }
    case Finger::kFinger3: {
      return "finger3";
    }
    default:
      throw std::runtime_error("Finger not valid.");
  }
}

int to_num(Finger finger) {
  switch (finger) {
    case Finger::kFinger1: {
      return 1;
      break;
    }
    case Finger::kFinger2: {
      return 2;
      break;
    }
    case Finger::kFinger3: {
      return 3;
      break;
    }
    default:
      throw std::runtime_error("Finger not valid.");
  }
}

Finger to_Finger(int i) {
  switch (i) {
    case 1:
      return Finger::kFinger1;
    case 2:
      return Finger::kFinger2;
    case 3:
      return Finger::kFinger3;
    default:
      throw std::runtime_error("Finger not valid");
  }
}

Finger to_Finger(std::string finger_name) {
  if (finger_name == "finger1") {
    return Finger::kFinger1;
  } else if (finger_name == "finger2") {
    return Finger::kFinger2;
  } else if (finger_name == "finger3") {
    return Finger::kFinger3;
  } else {
    throw std::runtime_error("Unknown finger name string");
  }
}

BrickFace to_BrickFace(std::string brick_face_name) {
  if (brick_face_name == "PosY") {
    return BrickFace::kPosY;
  } else if (brick_face_name == "NegY") {
    return BrickFace::kNegY;
  } else if (brick_face_name == "PosZ") {
    return BrickFace::kPosZ;
  } else if (brick_face_name == "NegZ") {
    return BrickFace::kNegZ;
  } else {
    throw std::runtime_error("Unknown brick face name");
  }
}

std::string to_string(BrickFace brick_face) {
  switch (brick_face) {
    case BrickFace::kPosY:
      return "PosY";
    case BrickFace::kNegY:
      return "NegY";
    case BrickFace::kPosZ:
      return "PosZ";
    case BrickFace::kNegZ:
      return "NegZ";
    default:
      throw std::runtime_error("BrickFace not valid.");
  }
}

template <typename T>
void WeldGripperFrames(MultibodyPlant<T>* plant, math::RigidTransformd X_WG) {
  // The finger base links are all welded a fixed distance from the gripper
  // frame's origin (Go), lying on the the gripper frame's Y-Z plane. We denote
  // The gripper frame's Y and Z axes as Gy and Gz.
  const double kGripperOriginToBaseDistance = 0.19;
  const double kFinger1Angle = FingerWeldAngle(Finger::kFinger1);
  const double kFinger2Angle = FingerWeldAngle(Finger::kFinger2);
  const double kFinger3Angle = FingerWeldAngle(Finger::kFinger3);

  // Note: Before welding and with all finger joint angles being zero, all
  // finger base links sit at the world origin with the finger pointing along
  // the world -Z axis.

  // Weld the first finger. Finger base links are arranged equidistant along the
  // perimeter of a circle. The first finger is welded kFinger1Angle radians
  // from the +Gz-axis. Frames F1, F2, F3 correspond to the base link finger
  // frames.
  RigidTransformd X_GF1 =
      RigidTransformd(Eigen::AngleAxisd(kFinger1Angle, Vector3d::UnitX()),
                      Vector3d(0, 0, 0)) *
      RigidTransformd(math::RotationMatrixd(),
                      Vector3d(0, 0, kGripperOriginToBaseDistance));
  const multibody::Frame<T>& finger1_base_frame =
      plant->GetFrameByName("finger1_base");
  plant->WeldFrames(plant->world_frame(), finger1_base_frame, X_WG * X_GF1);

  // Weld the second finger. The second finger is welded kFinger2Angle radians
  // from the +Gz-axis.
  RigidTransformd X_GF2 =
      RigidTransformd(Eigen::AngleAxisd(kFinger2Angle, Vector3d::UnitX()),
                      Vector3d(0, 0, 0)) *
      RigidTransformd(math::RotationMatrixd(),
                      Vector3d(0, 0, kGripperOriginToBaseDistance));
  const multibody::Frame<T>& finger2_base_frame =
      plant->GetFrameByName("finger2_base");
  plant->WeldFrames(plant->world_frame(), finger2_base_frame, X_WG * X_GF2);

  // Weld the 3rd finger. The third finger is welded kFinger3Angle radians from
  // the +Gz-axis.
  RigidTransformd X_GF3 =
      RigidTransformd(Eigen::AngleAxisd(kFinger3Angle, Vector3d::UnitX()),
                      Vector3d(0, 0, 0)) *
      RigidTransformd(math::RotationMatrixd(),
                      Vector3d(0, 0, kGripperOriginToBaseDistance));
  const multibody::Frame<T>& finger3_base_frame =
      plant->GetFrameByName("finger3_base");
  plant->WeldFrames(plant->world_frame(), finger3_base_frame, X_WG * X_GF3);
}

// Explicit instantiations.
template void WeldGripperFrames(MultibodyPlant<double>* plant,
                                math::RigidTransformd X_WG);

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

  // Set the brick's initial pose (expressed in the gripper frame G).
  if (brick_initial_pose != EigenPtr<Vector3<double>>(nullptr)) {
    (*brick_initial_pose)(0) = brick_joint_keyframes(0, 0);  // y-translate
    (*brick_initial_pose)(1) = brick_joint_keyframes(0, 1);  // z-translate
    (*brick_initial_pose)(2) = brick_joint_keyframes(0, 2);  // x-revolute
  }

  finger_joint_keyframes.transposeInPlace();

  // Create the finger joint name to row index map.
  std::map<std::string, int> finger_joint_name_to_row_index_map;
  for (size_t i = 0; i < finger_joint_ordering.size(); i++) {
    finger_joint_name_to_row_index_map[finger_joint_ordering[i]] = i;
  }

  return std::make_pair(finger_joint_keyframes,
                        finger_joint_name_to_row_index_map);
}

MatrixX<double> ReorderKeyframesForPlant(
    const MultibodyPlant<double>& plant, const MatrixX<double> keyframes,
    std::map<std::string, int>* finger_joint_name_to_row_index_map) {
  DRAKE_DEMAND(finger_joint_name_to_row_index_map != nullptr);
  if (static_cast<int>(finger_joint_name_to_row_index_map->size()) !=
      keyframes.rows()) {
    throw std::runtime_error(
        "The number of keyframe rows must match the size of "
        "finger_joint_name_to_row_index_map.");
  }
  if (keyframes.rows() != kNumGripperJoints) {
    throw std::runtime_error(
        "The number of keyframe rows must match the number of planar-gripper "
        "joints");
  }
  if (plant.num_positions() != kNumGripperJoints) {
    throw std::runtime_error(
        "The number of plant positions must exactly match the number of "
        "planar-gripper joints.");
  }
  std::map<std::string, int> original_map = *finger_joint_name_to_row_index_map;
  MatrixX<double> reordered_keyframes(keyframes);
  for (auto iter = original_map.begin(); iter != original_map.end(); ++iter) {
    auto joint_vel_start_index =
        plant.GetJointByName(iter->first).velocity_start();
    reordered_keyframes.row(joint_vel_start_index) =
        keyframes.row(iter->second);
    (*finger_joint_name_to_row_index_map)[iter->first] = joint_vel_start_index;
  }
  return reordered_keyframes;
}

double FingerWeldAngle(Finger finger) {
  switch (finger) {
    case Finger::kFinger1 :
      return M_PI / 3.0;
      break;
    case Finger::kFinger2 :
      return -M_PI / 3.0;
      break;
    case Finger::kFinger3 :
      return M_PI;
      break;
    default:
      throw std::logic_error("Unknown Finger");
  }
}

/// Utility to publish frames to LCM.
void PublishFramesToLcm(
    const std::string& channel_name,
    const std::unordered_map<std::string, RigidTransformd>& name_to_frame_map,
    drake::lcm::DrakeLcmInterface* lcm) {
  std::vector<RigidTransformd> poses;
  std::vector<std::string> names;
  for (const auto& pair : name_to_frame_map) {
    poses.push_back(pair.second);
    names.push_back(pair.first);
  }
  PublishFramesToLcm(channel_name, poses, names, lcm);
}

void PublishFramesToLcm(const std::string& channel_name,
                        const std::vector<RigidTransformd>& poses,
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
    math::RigidTransform<float> pose = poses[i].cast<float>();
    // Create a frame publisher
    Eigen::Vector3f goal_pos = pose.translation();
    Eigen::Quaternion<float> goal_quat =
        Eigen::Quaternion<float>(pose.rotation().matrix());
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

/// Publishes pre-defined body frames once.
void PublishBodyFrames(systems::Context<double>& plant_context,
                       const multibody::MultibodyPlant<double>& plant,
                       lcm::DrakeLcm& lcm, bool brick_only) {
  std::vector<std::string> body_names;
  std::vector<RigidTransformd> poses;

  // list the body names that we want to visualize.
  body_names.push_back("brick_link");

  if (!brick_only) {
    body_names.push_back("finger1_base");
    if (plant.HasBodyNamed("finger2_base")) {
      body_names.push_back("finger2_base");
    }
    if (plant.HasBodyNamed("finger3_base")) {
      body_names.push_back("finger3_base");
    }
  }

  for (size_t i = 0; i < body_names.size(); i++) {
    auto& body = plant.GetBodyByName(body_names[i]);
    math::RigidTransform<double> X_WB =
        plant.EvalBodyPoseInWorld(plant_context, body);
    poses.push_back(X_WB);
  }

  PublishFramesToLcm("SIM", poses, body_names, &lcm);
}

std::vector<std::string> GetPreferredGripperStateOrdering() {
  std::vector<std::string> user_order_vec;
  user_order_vec.push_back("finger1_BaseJoint");
  user_order_vec.push_back("finger1_MidJoint");
  user_order_vec.push_back("finger2_BaseJoint");
  user_order_vec.push_back("finger2_MidJoint");
  user_order_vec.push_back("finger3_BaseJoint");
  user_order_vec.push_back("finger3_MidJoint");
  return user_order_vec;
}

/// A system that publishes frames at a specified period.
FrameViz::FrameViz(const multibody::MultibodyPlant<double>& plant,
                   lcm::DrakeLcm& lcm, double period, bool frames_input)
    : plant_(plant), lcm_(lcm), frames_input_(frames_input) {
  this->DeclareVectorInputPort(
      "x", systems::BasicVector<double>(plant.num_positions() +
                                        plant.num_velocities() /* mbp state*/));
  // if true, then we create an additional input port which takes arbitrary
  // frames to visualize (a vector of type RigidTransform).
  if (frames_input_) {
    this->DeclareAbstractInputPort(Value<std::vector<math::RigidTransformd>>());
  }
  this->DeclarePeriodicPublishEvent(period, 0.,
                                    &FrameViz::PublishFramePose);
  plant_context_ = plant.CreateDefaultContext();
}

systems::EventStatus FrameViz::PublishFramePose(
    const drake::systems::Context<double>& context) const {
  auto state = this->EvalVectorInput(context, 0)->get_value();
  plant_.SetPositionsAndVelocities(plant_context_.get(), state);
  PublishBodyFrames(*plant_context_, plant_, lcm_, true /* brick only */);
  return systems::EventStatus::Succeeded();
}

/// Visualizes the spatial forces via Evan's spatial force visualization PR.
ExternalSpatialToSpatialViz::ExternalSpatialToSpatialViz(
    const MultibodyPlant<double>& plant, multibody::ModelInstanceIndex instance,
    double force_scale_factor)
    : plant_(plant),
      instance_(instance),
      force_scale_factor_(force_scale_factor) {
  // Make context with default parameters.
  plant_context_ = plant.CreateDefaultContext();
  this->DeclareAbstractInputPort(
      Value<std::vector<multibody::ExternallyAppliedSpatialForce<double>>>());
  this->DeclareVectorInputPort(
      "x", systems::BasicVector<double>(plant.num_multibody_states(
               instance) /* model_instance state size*/));

  // This output port produces a SpatialForceOutput, which feeds the spatial
  // forces visualization plugin of DrakeVisualizer.
  this->DeclareAbstractOutputPort(&ExternalSpatialToSpatialViz::CalcOutput);
}

// Computes the contact point in the world frame. Incoming spatial
// forces are already in the world frame.
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
    // Convert contact point from brick frame to world frame
    auto ext_spatial_force = external_spatial_forces_vec[i];
    //     drake::log()->info("p: \n{}", ext_spatial_force.p_BoBq_B);
    //     drake::log()->info("f: \n{}",
    //     ext_spatial_force.F_Bq_W.translational());

    auto& body = plant_.get_body(ext_spatial_force.body_index);
    auto& X_WB = plant_.EvalBodyPoseInWorld(*plant_context_, body);
    auto p_BoBq_W = X_WB * ext_spatial_force.p_BoBq_B;

    spatial_forces_viz_output->emplace_back(
        p_BoBq_W, ext_spatial_force.F_Bq_W * force_scale_factor_);
  }
}

MapStateToUserOrderedState::MapStateToUserOrderedState(
    const MultibodyPlant<double>& plant,
    std::vector<std::string> user_order_vec) {
  // Create the state selctor matrix.
  std::vector<multibody::JointIndex> joint_indices;
  for (auto iter = user_order_vec.begin(); iter != user_order_vec.end();
       ++iter) {
    joint_indices.push_back(plant.GetJointByName(*iter).index());
  }
  Sx_ = plant.MakeStateSelectorMatrix(joint_indices);

  this->DeclareVectorInputPort(
      "plant_state", systems::BasicVector<double>(plant.num_positions() +
                                                  plant.num_velocities()));

  this->DeclareVectorOutputPort(
      "user_x", systems::BasicVector<double>(user_order_vec.size() * 2),
      &MapStateToUserOrderedState::CalcOutput);
}

void MapStateToUserOrderedState::CalcOutput(
    const systems::Context<double>& context,
    systems::BasicVector<double>* output_vector) const {
  auto output_value = output_vector->get_mutable_value();
  auto plant_state = this->EvalVectorInput(context, 0)->get_value();

  output_value.setZero();
  output_value = Sx_ * plant_state;  // User ordered state.
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
