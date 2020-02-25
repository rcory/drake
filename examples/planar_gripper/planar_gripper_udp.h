#pragma once

/// @file
/// This file contains classes dealing with sending/receiving UDP messages
/// related to the planar gripper.

#include "drake/examples/planar_gripper/planar_gripper_common.h"

namespace drake {
namespace examples {
namespace planar_gripper {
struct UdpMessage {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(UdpMessage)

  UdpMessage() {}

  virtual ~UdpMessage() {}

  int GetMessageSize() const { return DoGetMessageSize(); }

  void Deserialize(const uint8_t* msg) { DoDeserialize(msg); }
  void Serialize(uint8_t* msg) const { DoSerialize(msg); }

 private:
  virtual int DoGetMessageSize() const = 0;

  virtual void DoDeserialize(const uint8_t* msg) = 0;

  virtual void DoSerialize(uint8_t* msg) const = 0;
};

struct FingerFaceAssignment : public UdpMessage {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FingerFaceAssignment)

  FingerFaceAssignment() {}

  virtual ~FingerFaceAssignment() {}

  uint32_t utime;

  Finger finger;
  BrickFace brick_face;

  Eigen::Vector2d p_BoBq_B;

 private:
  virtual int DoGetMessageSize() const final;

  virtual void DoDeserialize(const uint8_t* msg) final;

  virtual void DoSerialize(uint8_t* msg) const final;
};

bool operator==(const FingerFaceAssignment& f1, const FingerFaceAssignment& f2);

struct FingerFaceAssignments : public UdpMessage {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FingerFaceAssignments)

  FingerFaceAssignments(int m_num_fingers)
      : num_fingers{static_cast<uint32_t>(m_num_fingers)},
        finger_face_assignments(m_num_fingers) {}

  FingerFaceAssignments() : FingerFaceAssignments(0) {}

  virtual ~FingerFaceAssignments() {}

  uint32_t utime;

  uint32_t num_fingers;
  std::vector<FingerFaceAssignment> finger_face_assignments;

 private:
  virtual int DoGetMessageSize() const final;

  virtual void DoDeserialize(const uint8_t* msg) final;

  virtual void DoSerialize(uint8_t* msg) const final;
};

bool operator==(const FingerFaceAssignments& f1,
                const FingerFaceAssignments& f2);

struct PlanarManipulandDesired : public UdpMessage {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(PlanarManipulandDesired)
  PlanarManipulandDesired(int m_num_states, int m_num_accel)
      : num_states{static_cast<uint32_t>(m_num_states)},
        num_accels{static_cast<uint32_t>(m_num_accel)},
        desired_state(num_states),
        desired_accel(num_accels) {}

  ~PlanarManipulandDesired() {}

  uint32_t utime;
  uint32_t num_states;
  uint32_t num_accels;
  Eigen::VectorXd desired_state;
  Eigen::VectorXd desired_accel;

 private:
  virtual int DoGetMessageSize() const final;

  virtual void DoDeserialize(const uint8_t* msg) final;

  virtual void DoSerialize(uint8_t* msg) const final;
};

bool operator==(const PlanarManipulandDesired& f1,
                const PlanarManipulandDesired& f2);

struct PlanarManipulandSpatialForce : public UdpMessage {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(PlanarManipulandSpatialForce)

  PlanarManipulandSpatialForce() {}

  virtual ~PlanarManipulandSpatialForce() {}

  uint32_t utime;
  Finger finger;
  // (y, z) position of point Bq in body frame B.
  Eigen::Vector2d p_BoBq_B;

  // (y, z) force applied to body B at point Bq, expressed in the world frame W.
  Eigen::Vector2d force_Bq_W;

  // torque applied to body B at point Bq, expressed in the world frame W.
  double torque_Bq_W;

 private:
  virtual int DoGetMessageSize() const final;

  virtual void DoDeserialize(const uint8_t* msg) final;

  virtual void DoSerialize(uint8_t* msg) const final;
};

bool operator==(const PlanarManipulandSpatialForce& f1,
                const PlanarManipulandSpatialForce& f2);

struct PlanarManipulandSpatialForces : public UdpMessage {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(PlanarManipulandSpatialForces)

  PlanarManipulandSpatialForces(int m_num_forces);

  PlanarManipulandSpatialForces();

  ~PlanarManipulandSpatialForces() {}

  uint32_t utime;

  uint32_t num_forces;
  std::vector<PlanarManipulandSpatialForce> forces;

 private:
  virtual int DoGetMessageSize() const final;

  virtual void DoDeserialize(const uint8_t* msg) final;

  virtual void DoSerialize(uint8_t* msg) const final;
};

bool operator==(const PlanarManipulandSpatialForces& f1,
                const PlanarManipulandSpatialForces& f2);

struct PlanarPlantState : public UdpMessage {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(PlanarPlantState)

  PlanarPlantState(int m_num_states);

  PlanarPlantState();

  virtual ~PlanarPlantState() {}

  uint32_t utime;
  uint32_t num_states;
  Eigen::VectorXd plant_state;

 private:
  virtual int DoGetMessageSize() const final;

  virtual void DoDeserialize(const uint8_t* msg) final;

  virtual void DoSerialize(uint8_t* msg) const final;
};

bool operator==(const PlanarPlantState& f1, const PlanarPlantState& f2);
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
