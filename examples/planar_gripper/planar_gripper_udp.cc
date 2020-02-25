#include "drake/examples/planar_gripper/planar_gripper_udp.h"

namespace drake {
namespace examples {
namespace planar_gripper {
int FingerFaceAssignment::DoGetMessageSize() const {
  return sizeof(uint32_t) + sizeof(Finger) + sizeof(BrickFace) +
         sizeof(double) * 2;
}

namespace {
template <typename T>
void DeserializeBytes(T* dst, const uint8_t* msg, int num_bytes, int* start) {
  // Copy the @p num_bytes bytes from msg + *start to @p dst, and then increment
  // @p start by size.
  memcpy(dst, msg + *start, num_bytes);
  *start += num_bytes;
}

template <typename T>
void SerializeBytes(uint8_t* msg, int num_bytes, T* source, int* start) {
  // Copy *source to msg + *start, and occupy the next @p num_bytes bytes.
  // Also increment *start by @p num_bytes.
  memcpy(msg + *start, source, num_bytes);
  *start += num_bytes;
}
}  // namespace

void FingerFaceAssignment::DoDeserialize(const uint8_t* msg) {
  int start = 0;
  DeserializeBytes(&(this->utime), msg, sizeof(uint32_t), &start);
  DeserializeBytes(&(this->finger), msg, sizeof(Finger), &start);
  DeserializeBytes(&(this->brick_face), msg, sizeof(BrickFace), &start);
  DeserializeBytes(p_BoBq_B.data(), msg, sizeof(double) * 2, &start);
}

void FingerFaceAssignment::DoSerialize(uint8_t* msg) const {
  int start = 0;
  SerializeBytes(msg, sizeof(uint32_t), &(this->utime), &start);
  SerializeBytes(msg, sizeof(Finger), &(this->finger), &start);
  SerializeBytes(msg, sizeof(BrickFace), &(this->brick_face), &start);
  SerializeBytes(msg, sizeof(double) * 2, this->p_BoBq_B.data(), &start);
}

bool operator==(const FingerFaceAssignment& f1,
                const FingerFaceAssignment& f2) {
  return f1.utime == f2.utime && f1.finger == f2.finger &&
         f1.brick_face == f2.brick_face && f1.p_BoBq_B == f2.p_BoBq_B;
}

int FingerFaceAssignments::DoGetMessageSize() const {
  int size = sizeof(uint32_t) + sizeof(uint32_t);
  for (const auto& finger_face_assignment : finger_face_assignments) {
    size += finger_face_assignment.GetMessageSize();
  }
  return size;
}

void FingerFaceAssignments::DoDeserialize(const uint8_t* msg) {
  int start = 0;
  DeserializeBytes(&(this->utime), msg, sizeof(uint32_t), &start);
  DeserializeBytes(&(this->num_fingers), msg, sizeof(uint32_t), &start);
  this->finger_face_assignments.resize(this->num_fingers);
  for (int i = 0; i < static_cast<int>(this->num_fingers); ++i) {
    finger_face_assignments[i].Deserialize(msg + start);
    start += finger_face_assignments[i].GetMessageSize();
  }
}

void FingerFaceAssignments::DoSerialize(uint8_t* msg) const {
  int start = 0;
  SerializeBytes(msg, sizeof(uint32_t), &(this->utime), &start);
  SerializeBytes(msg, sizeof(uint32_t), &(this->num_fingers), &start);
  for (int i = 0; i < static_cast<int>(num_fingers); ++i) {
    this->finger_face_assignments[i].Serialize(msg + start);
    start += this->finger_face_assignments[i].GetMessageSize();
  }
}

bool operator==(const FingerFaceAssignments& f1,
                const FingerFaceAssignments& f2) {
  bool equal = f1.utime == f2.utime && f1.num_fingers == f2.num_fingers;
  if (!equal) {
    return false;
  }
  for (int i = 0; i < static_cast<int>(f1.num_fingers); ++i) {
    equal = equal &&
            (f1.finger_face_assignments[i] == f2.finger_face_assignments[i]);
  }
  return equal;
}

int PlanarManipulandDesired::DoGetMessageSize() const {
  return sizeof(uint32_t) * 3 + sizeof(double) * (num_states + num_accels);
}

void PlanarManipulandDesired::DoDeserialize(const uint8_t* msg) {
  int start = 0;
  DeserializeBytes(&(this->utime), msg, sizeof(uint32_t), &start);
  DeserializeBytes(&(this->num_states), msg, sizeof(uint32_t), &start);
  DeserializeBytes(&(this->num_accels), msg, sizeof(uint32_t), &start);
  this->desired_state.resize(this->num_states);
  DeserializeBytes(this->desired_state.data(), msg, sizeof(double) * num_states,
                   &start);
  this->desired_accel.resize(this->num_accels);
  DeserializeBytes(this->desired_accel.data(), msg, sizeof(double) * num_accels,
                   &start);
}

void PlanarManipulandDesired::DoSerialize(uint8_t* msg) const {
  int start = 0;
  SerializeBytes(msg, sizeof(uint32_t), &(this->utime), &start);
  SerializeBytes(msg, sizeof(uint32_t), &(this->num_states), &start);
  SerializeBytes(msg, sizeof(uint32_t), &(this->num_accels), &start);
  SerializeBytes(msg, sizeof(double) * num_states, this->desired_state.data(),
                 &start);
  SerializeBytes(msg, sizeof(double) * num_accels, this->desired_accel.data(),
                 &start);
}

bool operator==(const PlanarManipulandDesired& f1,
                const PlanarManipulandDesired& f2) {
  return f1.utime == f2.utime && f1.num_states == f2.num_states &&
         f1.num_accels == f2.num_accels &&
         f1.desired_state == f2.desired_state &&
         f1.desired_accel == f2.desired_accel;
}

int PlanarManipulandSpatialForce::DoGetMessageSize() const {
  return sizeof(uint32_t) + sizeof(Finger) + sizeof(double) * 5;
}

void PlanarManipulandSpatialForce::DoDeserialize(const uint8_t* msg) {
  int start = 0;
  DeserializeBytes(&(this->utime), msg, sizeof(uint32_t), &start);
  DeserializeBytes(&(this->finger), msg, sizeof(Finger), &start);
  DeserializeBytes(this->p_BoBq_B.data(), msg, sizeof(double) * 2, &start);
  DeserializeBytes(this->force_Bq_W.data(), msg, sizeof(double) * 2, &start);
  DeserializeBytes(&(this->torque_Bq_W), msg, sizeof(double), &start);
}

void PlanarManipulandSpatialForce::DoSerialize(uint8_t* msg) const {
  int start = 0;
  SerializeBytes(msg, sizeof(uint32_t), &(this->utime), &start);
  SerializeBytes(msg, sizeof(Finger), &(this->finger), &start);
  SerializeBytes(msg, sizeof(double) * 2, this->p_BoBq_B.data(), &start);
  SerializeBytes(msg, sizeof(double) * 2, this->force_Bq_W.data(), &start);
  SerializeBytes(msg, sizeof(double), &(this->torque_Bq_W), &start);
}

bool operator==(const PlanarManipulandSpatialForce& f1,
                const PlanarManipulandSpatialForce& f2) {
  return f1.utime == f2.utime && f1.finger == f2.finger &&
         f1.p_BoBq_B == f2.p_BoBq_B && f1.force_Bq_W == f2.force_Bq_W &&
         f1.torque_Bq_W == f2.torque_Bq_W;
}

PlanarManipulandSpatialForces::PlanarManipulandSpatialForces(int m_num_forces)
    : num_forces{static_cast<uint32_t>(m_num_forces)}, forces{num_forces} {};

PlanarManipulandSpatialForces::PlanarManipulandSpatialForces()
    : PlanarManipulandSpatialForces(0) {}

int PlanarManipulandSpatialForces::DoGetMessageSize() const {
  int size = sizeof(uint32_t) * 2;
  for (const auto& force : forces) {
    size += force.GetMessageSize();
  }
  return size;
}

void PlanarManipulandSpatialForces::DoDeserialize(const uint8_t* msg) {
  int start = 0;
  DeserializeBytes(&(this->utime), msg, sizeof(uint32_t), &start);
  DeserializeBytes(&(this->num_forces), msg, sizeof(uint32_t), &start);
  this->forces.resize(this->num_forces);
  for (int i = 0; i < static_cast<int>(this->num_forces); ++i) {
    forces[i].Deserialize(msg + start);
    start += forces[i].GetMessageSize();
  }
}

void PlanarManipulandSpatialForces::DoSerialize(uint8_t* msg) const {
  int start = 0;
  SerializeBytes(msg, sizeof(uint32_t), &(this->utime), &start);
  SerializeBytes(msg, sizeof(uint32_t), &(this->num_forces), &start);
  for (int i = 0; i < static_cast<int>(num_forces); ++i) {
    this->forces[i].Serialize(msg + start);
    start += this->forces[i].GetMessageSize();
  }
}

bool operator==(const PlanarManipulandSpatialForces& f1,
                const PlanarManipulandSpatialForces& f2) {
  bool result = f1.utime == f2.utime && f1.num_forces == f2.num_forces;
  if (result == false) {
    return false;
  }
  for (int i = 0; i < static_cast<int>(f1.num_forces); ++i) {
    result = result && f1.forces[i] == f2.forces[i];
  }
  return result;
}

PlanarPlantState::PlanarPlantState(int m_num_states)
    : num_states(m_num_states), plant_state(num_states) {}

PlanarPlantState::PlanarPlantState() : PlanarPlantState(0) {}

int PlanarPlantState::DoGetMessageSize() const {
  return sizeof(uint32_t) * 2 + sizeof(double) * num_states;
}

void PlanarPlantState::DoDeserialize(const uint8_t* msg) {
  int start = 0;
  DeserializeBytes(&(this->utime), msg, sizeof(uint32_t), &start);
  DeserializeBytes(&(this->num_states), msg, sizeof(uint32_t), &start);
  this->plant_state.resize(this->num_states);
  DeserializeBytes(this->plant_state.data(), msg,
                   sizeof(double) * this->num_states, &start);
}

void PlanarPlantState::DoSerialize(uint8_t* msg) const {
  int start = 0;
  SerializeBytes(msg, sizeof(uint32_t), &(this->utime), &start);
  SerializeBytes(msg, sizeof(uint32_t), &(this->num_states), &start);
  SerializeBytes(msg, sizeof(double) * this->num_states,
                 this->plant_state.data(), &start);
}

bool operator==(const PlanarPlantState& f1, const PlanarPlantState& f2) {
  return f1.utime == f2.utime && f1.num_states == f2.num_states &&
         f1.plant_state == f2.plant_state;
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
