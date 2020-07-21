#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/spatial_force_output.h"

namespace drake {
namespace examples {
namespace planar_gripper {

using drake::multibody::MultibodyPlant;
using Eigen::Vector3d;

// TODO(rcory) These values should be moved to PlanarGripper class.
constexpr int kNumFingers = 3;
constexpr int kNumJointsPerFinger = 2;
constexpr int kNumGripperJoints = kNumFingers * kNumJointsPerFinger;

enum class Finger {
  kFinger1 = 1,
  kFinger2,
  kFinger3,
};
std::string to_string(Finger finger);
std::string to_string_from_finger_num(int i);
int to_num(Finger finger);
Finger to_Finger(int i);
Finger to_Finger(std::string finger_name);

enum class BrickFace {
  kPosZ,
  kNegZ,
  kPosY,
  kNegY,
  kClosest,  // Force controller chooses the closest face/point.
};
BrickFace to_BrickFace(std::string brick_face_name);
std::string to_string(BrickFace brick_face);

// The planar-gripper coordinate frame G (with origin Go) and finger layout are
// defined as follows (assuming all finger joint angles are set to zero):
//
//       F1_base         F2_base
//              \   +Gz    /
//               \   |   /
//                \  |  /
//                 ● | ●
//                   Go----+Gy
//                   ●
//                   |
//                   |
//                   |
//                 F3_base
//
// The gripper frame's Y and Z axes are denote Gy and Gz, respectively. When the
// planar-gripper is welded via WeldGripperFrames(), the coordinate frame G
// perfectly coincides with the world coordinate frame W.

/**
 * Welds each finger's base frame to the world. The planar-gripper is made up of
 * three 2-DOF fingers, whose bases are fixed equidistant along the perimeter of
 * a circle. The origin (Go) of the planar gripper is located at the center of
 * the workspace, i.e., if all joints are set to zero and all fingers are
 * pointing inwards, the origin is the point that is equidistant to all
 * fingertips. This method welds the planar-gripper such that all motion lies in
 * the Y-Z plane (in frame G). Note: The planar gripper frame G perfectly
 * coincides with the world coordinate frame W when welded via this method.
 * @tparam T The scalar type. Currently only supports double.
 * @param plant The plant containing the planar-gripper.
 * @param X_WG A RigidTransform containing the pose of the gripper frame (G)
 * w.r.t. the world.
 * @tparam_double_only
 */
template <typename T>
void WeldGripperFrames(MultibodyPlant<T>* plant,
                       math::RigidTransformd X_WG = math::RigidTransformd());

/**
 * Parses a text file containing keyframe joint positions for the planar gripper
 * and the planar brick (the object being manipulated).
 * @param[in] name The file name to parse.
 * @param[out] times A vector that will contain the time stamps of the
 * keyframes.
 * @param[out] modes A matrix that will contain the contact modes for each
 * finger. This matrix will be of size kNumFingers x n, where n is the number
 * of keyframes. The first row of this matrix corresponds to finger1, the second
 * row to finger 2, and so on.
 * @param[out] brick_keyframe_info A std::pair containing a matrix of brick
 * joint position keyframes (each matrix column represents a single keyframe
 * containing values for all joint positions) and a std::map containing the
 * mapping between each brick joint name and the corresponding row index in the
 * keyframe matrix containing the data for that joint. Values are expressed in
 * the gripper frame G.
 * @return A std::pair containing a matrix of finger joint position keyframes
 * (each matrix column represents a single keyframe containing values for all
 * joint positions) and a std::map containing the mapping between each finger
 * joint name and the corresponding row index in the keyframe matrix containing
 * the data for that joint.
 * @pre The file should begin with a header row that indicates the joint
 * ordering for keyframes. Header names should consist of three finger base
 * joints, three finger mid joints, and three brick joints (9 total):
 * {finger1_BaseJoint, finger2_BaseJoint, finger3_BaseJoint, finger1_MidJoint,
 * finger2_MidJoint, finger3_MindJoint, brick_translate_y_joint,
 * brick_translate_z_joint, brick_revolute_x_joint}. Note that brick
 * translations should be expressed in the planar-gripper frame G. Names may
 * appear in any order.
 * After the header row, a repeating sequence of rows should be found. The first
 * row corresponds to a plant keyframe, the second row corresponds to the time
 * stamp of that keyframe, and the third row corresponds to the finger contact
 * modes, one value for each finger. The first value in this row is the contact
 * mode for finger1, the second value is for finger 2, and so on. From the
 * r3t_dexterity documentation:
 * For each of the fingers, we record the index of the face in contact with that
 * finger. The face is 0-indexed, with face i being the face connecting
 * PlanarObject.vertices[i] and PlanarObject.vertices[i+1]. When the finger is
 * not in contact with any face, we use -1 as the face index in the plan txt
 * output file. So a list [0, 2, -1] means that finger 1 is in contact with
 * face 0, finger 2 is in contact with face 2, and finger 3 is not in contact.
 * The index of the vertices could be accessed through python in
 * planar_gripper_system.manipuland.vertices. For the brick case, edge 0 is the
 * left (-y) edge, edge 1 is the top (+z) edge, edge 2 is the right (+y) edge,
 * and edge 3 is the bottom (-z) edge.
 *
 * Each keyframe row should contain the same number of values as indicated in
 * the header. All entries should be white space delimited and the file should
 * end in a newline character. The behavior of parsing is undefined if these
 * conditions are not met.
 */
std::pair<MatrixX<double>, std::map<std::string, int>> ParseKeyframesAndModes(
    const std::string& name, VectorX<double>* times, MatrixX<double>* modes,
    std::pair<MatrixX<double>, std::map<std::string, int>>*
        brick_keyframe_info = nullptr);

/**
 * Reorders the joint keyframe matrix data contained in `keyframes` such that
 * joint keyframes (rows) are ordered according to the `plant`'s joint velocity
 * index ordering. This is useful, for example, in making the keyframe matrix
 * data compatible with the inverse dynamics controller's desired state input
 * port ordering. In this case, the incoming `plant` is the MultibodyPlant used
 * for inverse dynamics control, i.e., the "control plant". The number of
 * planar-gripper joints `kNumJoints` must exactly match plant.num_positions().
 * @param[in] plant The MultibodyPlant providing the velocity index ordering.
 * @param[in] keyframes The keyframes data.
 * @param[out] joint_name_to_row_index_map A std::map which contains the
 * incoming joint name to row index ordering. This map is updated to reflect the
 * new keyframe reordering.
 * @return A MatrixX containing the reordered keyframes.
 * @throw If the number of keyframe rows does not match the size of
 * `joint_name_to_row_index_map`
 * @throw If the number of keyframe rows does not match the number of
 * planar-gripper joints.
 * @throw If the number of keyframe rows does not exactly match
 * plant.num_positions().
 */
MatrixX<double> ReorderKeyframesForPlant(
    const MultibodyPlant<double>& plant, const MatrixX<double> keyframes,
    std::map<std::string, int>* joint_name_to_row_index_map);

/// Returns a specific finger's weld angle from the +Gz axis
/// (gripper frame, +z axis)
double FingerWeldAngle(Finger finger);

/// Utility to publish frames to LCM.
void PublishFramesToLcm(
    const std::string& channel_name,
    const std::unordered_map<std::string, math::RigidTransformd>&
        name_to_frame_map,
    drake::lcm::DrakeLcmInterface* lcm);

void PublishFramesToLcm(const std::string& channel_name,
                        const std::vector<math::RigidTransformd>& frames,
                        const std::vector<std::string>& frame_names,
                        drake::lcm::DrakeLcmInterface* lcm);

/// Publishes pre-defined body frames once.
void PublishBodyFrames(const systems::Context<double>& plant_context,
                       const multibody::MultibodyPlant<double>& plant,
                       lcm::DrakeLcm* lcm);

/// Returns the preferred joint ordering for an individual finger.
std::vector<std::string> GetPreferredFingerJointOrdering();

/// Returns the preferred joint ordering for the planar gripper only (e.g.,
/// used to create the state selector matrix using MBP). The preferred
/// ordering defined here ensures joint ordering is by finger number, and each
/// finger is ordered according to GetPreferredFingerJointOrdering().
std::vector<std::string> GetPreferredGripperJointOrdering();

/// Returns the plant's state selector matrix corresponding to the
/// provided joint names in PlanarGripper.
MatrixX<double> MakeStateSelectorMatrix(
    const MultibodyPlant<double>& plant,
    const std::vector<std::string>& joint_names);

/// A system that publishes frames at a specified period.
class FrameViz final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FrameViz)

  FrameViz(const multibody::MultibodyPlant<double>& plant, lcm::DrakeLcm* lcm,
           double period, bool frames_input = false);

 private:
  systems::EventStatus PublishFramePose(
      const systems::Context<double>& context) const;

  const multibody::MultibodyPlant<double>& plant_;
  std::unique_ptr<systems::Context<double>> plant_context_;
  lcm::DrakeLcm* lcm_;
  bool frames_input_{false};
};

/// Visualizes the spatial forces via Evan's spatial force visualization PR.
/// A system that takes an `ExternallyAppliedSpatialForce` as input, and
/// outputs a std::vector of type `SpatialForceOutput`, which is then used for
/// visualization. The latter omits the body index, and expresses the contact
/// point in the world frame, instead of the body frame. The force is expressed
/// in the world frame for both.
class ExternalSpatialToSpatialViz final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ExternalSpatialToSpatialViz)

  ExternalSpatialToSpatialViz(const multibody::MultibodyPlant<double>& plant,
                              multibody::ModelInstanceIndex instance,
                              double force_scale_factor = 10);

  void CalcOutput(const systems::Context<double>& context,
                  std::vector<multibody::SpatialForceOutput<double>>*
                      spatial_forces_viz_output) const;

 private:
  const multibody::MultibodyPlant<double>& plant_;
  multibody::ModelInstanceIndex instance_;
  std::unique_ptr<systems::Context<double>> plant_context_;
  double force_scale_factor_;
};

/// Takes in a state vector from MBP state output port, and outputs a state
/// whose order is dictated by `user_order`. The user ordered output state can
/// be a subset of the full state.
class MapPlantStateToUserOrderedState final
    : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MapPlantStateToUserOrderedState)

  MapPlantStateToUserOrderedState(const MultibodyPlant<double>& plant,
                                  std::vector<std::string> user_order_vec);

  void CalcOutput(const systems::Context<double>& context,
                  systems::BasicVector<double>* output_vector) const;

 private:
  MatrixX<double> Sx_;  // state selector matrix.
};

/// Takes in a plant state vector whose ordering is given by joint names
/// contained in `user_order_vec` and outputs a state vector whose ordering
/// matches the MBP joint ordering. The user ordered input state must be the
/// same size as MultibodyPlant::num_velocities(model_index). If the optional
/// argument model index is not provided, this size is equal to the number of
/// velocities in the full MBP.
class MapUserOrderedStateToPlantState final
    : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MapUserOrderedStateToPlantState)

  MapUserOrderedStateToPlantState(
      const MultibodyPlant<double>& plant,
      const std::vector<std::string>& user_order_vec,
      std::optional<multibody::ModelInstanceIndex> model_index);

  void CalcOutput(const systems::Context<double>& context,
                  systems::BasicVector<double>* output_vector) const;

 private:
  MatrixX<double> Sx_inv_;  // state selector matrix inverse.
};

/// Takes in a plant actuation vector whose ordering is given by joint names
/// contained in `user_order_vec` and outputs an actuation vector whose ordering
/// matches the MBP actuation ordering. The user ordered input must be the same
/// size as MultibodyPlant::num_actuated_dofs(model_index). If the optional
/// argument `model_index` is not provided, this size is equal to the number of
/// actuated dofs in the full MBP.
class MapUserOrderedActuationToPlantActuation final
    : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MapUserOrderedActuationToPlantActuation)

  MapUserOrderedActuationToPlantActuation(
      const MultibodyPlant<double>& plant,
      const std::vector<std::string>& user_order_vec,
      std::optional<multibody::ModelInstanceIndex> model_index);

  void CalcOutput(const systems::Context<double>& context,
                  systems::BasicVector<double>* output_vector) const;

 private:
  MatrixX<double> Su_inv_;  // actuation selector matrix inverse.
};

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
