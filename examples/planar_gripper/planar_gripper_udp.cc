#include "drake/examples/planar_gripper/planar_gripper_udp.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <memory>

#include "drake/systems/framework/abstract_values.h"

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
  int size = sizeof(uint32_t) + sizeof(uint32_t) + sizeof(bool) * num_fingers;
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
  this->in_contact.resize(this->num_fingers);
  for (int i = 0; i < static_cast<int>(this->num_fingers); ++i) {
    bool flag;
    memcpy(&flag, msg + start, sizeof(bool));
    in_contact[i] = flag;
    start += sizeof(bool);
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
  for (int i = 0; i < static_cast<int>(this->num_fingers); ++i) {
    bool flag = in_contact[i];
    memcpy(msg + start, &flag, sizeof(bool));
    start += sizeof(bool);
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
    equal = equal && f1.in_contact[i] == f2.in_contact[i];
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

multibody::ExternallyAppliedSpatialForce<double>
PlanarManipulandSpatialForce::ToSpatialForce(
    multibody::BodyIndex body_index) const {
  multibody::ExternallyAppliedSpatialForce<double> applied_spatial_force;
  applied_spatial_force.body_index = body_index;
  applied_spatial_force.p_BoBq_B(0) = 0;  // x
  applied_spatial_force.p_BoBq_B.tail<2>() = p_BoBq_B;
  Eigen::Vector3d force;
  force << 0, force_Bq_W;
  Eigen::Vector3d torque = Eigen::Vector3d::Zero();
  torque(0) = torque_Bq_W;  // tau_x
  applied_spatial_force.F_Bq_W = multibody::SpatialForce<double>(torque, force);
  return applied_spatial_force;
}

void PlanarManipulandSpatialForce::FromSpatialForce(
    const multibody::ExternallyAppliedSpatialForce<double>& spatial_force) {
  this->p_BoBq_B = spatial_force.p_BoBq_B.tail<2>();
  this->force_Bq_W = spatial_force.F_Bq_W.translational().tail<2>();
  this->torque_Bq_W = spatial_force.F_Bq_W.rotational()(0);
}

bool operator==(const PlanarManipulandSpatialForce& f1,
                const PlanarManipulandSpatialForce& f2) {
  return f1.utime == f2.utime && f1.finger == f2.finger &&
         f1.p_BoBq_B == f2.p_BoBq_B && f1.force_Bq_W == f2.force_Bq_W &&
         f1.torque_Bq_W == f2.torque_Bq_W;
}

PlanarManipulandSpatialForces::PlanarManipulandSpatialForces(int m_num_forces)
    : num_forces{static_cast<uint32_t>(m_num_forces)},
      forces{num_forces},
      in_contact(m_num_forces, false) {}

PlanarManipulandSpatialForces::PlanarManipulandSpatialForces()
    : PlanarManipulandSpatialForces(0) {}

int PlanarManipulandSpatialForces::DoGetMessageSize() const {
  int size = sizeof(uint32_t) * 2 + sizeof(bool) * num_forces;
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
  this->in_contact.resize(this->num_forces);
  for (int i = 0; i < static_cast<int>(this->num_forces); ++i) {
    bool flag;
    memcpy(&flag, msg + start, sizeof(bool));
    in_contact[i] = flag;
    start += sizeof(bool);
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
  for (int i = 0; i < static_cast<int>(this->num_forces); ++i) {
    bool flag = in_contact[i];
    memcpy(msg + start, &flag, sizeof(bool));
    start += sizeof(bool);
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
    result = result && f1.in_contact[i] == f2.in_contact[i];
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

SimToQPUdpPublisherSystem::SimToQPUdpPublisherSystem(
    double publish_period, int local_port, int remote_port,
    uint32_t remote_address, int num_plant_states, int num_fingers,
    int num_brick_states, int num_brick_accels)
    : file_descriptor_{socket(AF_INET, SOCK_DGRAM, 0)},
      remote_port_{remote_port},
      remote_address_{remote_address},
      num_plant_states_{num_plant_states},
      num_fingers_{num_fingers},
      num_brick_states_{num_brick_states},
      num_brick_accels_{num_brick_accels} {
  struct sockaddr_in myaddr;
  myaddr.sin_family = AF_INET;
  myaddr.sin_addr.s_addr = htonl(INADDR_ANY);
  myaddr.sin_port = htons(local_port);
  int status =
      bind(file_descriptor_, reinterpret_cast<struct sockaddr*>(&myaddr),
           sizeof(myaddr));
  if (status < 0) {
    throw std::runtime_error("Cannot bind the UDP file descriptor.");
  }

  this->DeclareForcedPublishEvent(
      &SimToQPUdpPublisherSystem::PublishInputAsUdpMessage);

  const double offset = 0.0;
  this->DeclarePeriodicPublishEvent(
      publish_period, offset,
      &SimToQPUdpPublisherSystem::PublishInputAsUdpMessage);

  plant_state_input_port_ =
      this->DeclareVectorInputPort(
              "qp_estimated_plant_state",
              systems::BasicVector<double>(num_plant_states))
          .get_index();

  finger_face_assignments_input_port_ =
      this->DeclareAbstractInputPort(
              "qp_finger_face_assignments",
              Value<std::unordered_map<
                  Finger, std::pair<BrickFace, Eigen::Vector2d>>>{})
          .get_index();

  desired_brick_state_input_port_ =
      this->DeclareVectorInputPort(
              "qp_desired_brick_state",
              systems::BasicVector<double>(num_brick_states_))
          .get_index();
  desired_brick_accel_input_port_ =
      this->DeclareVectorInputPort(
              "qp_desired_brick_accel",
              systems::BasicVector<double>(num_brick_accels_))
          .get_index();
}

std::vector<uint8_t> SimToQPUdpPublisherSystem::Serialize(
    const systems::Context<double>& context) const {
  // Construct PlanarPlantState
  PlanarPlantState plant_state(num_plant_states_);
  const uint32_t utime = context.get_time() * 1e6;
  plant_state.utime = utime;
  const systems::BasicVector<double>* plant_state_input =
      this->EvalVectorInput(context, plant_state_input_port_);
  plant_state.plant_state = plant_state_input->CopyToVector();

  // Construct FingerFaceAssignments.
  FingerFaceAssignments finger_face_assignments(num_fingers_);
  finger_face_assignments.utime = utime;
  const auto finger_face_assignments_input =
      this->get_input_port(finger_face_assignments_input_port_)
          .Eval<std::unordered_map<Finger,
                                   std::pair<BrickFace, Eigen::Vector2d>>>(
              context);
  int finger_face_index = 0;
  for (const auto& finger_face_assignment : finger_face_assignments_input) {
    finger_face_assignments.finger_face_assignments[finger_face_index].utime =
        utime;
    finger_face_assignments.finger_face_assignments[finger_face_index].finger =
        finger_face_assignment.first;
    finger_face_assignments.finger_face_assignments[finger_face_index]
        .brick_face = finger_face_assignment.second.first;
    finger_face_assignments.finger_face_assignments[finger_face_index]
        .p_BoBq_B = finger_face_assignment.second.second;
    finger_face_assignments.in_contact[finger_face_index] = true;
    finger_face_index++;
  }

  // Construct PlanarManipulandDesired
  PlanarManipulandDesired planar_manipuland_desired(num_brick_states_,
                                                    num_brick_accels_);
  planar_manipuland_desired.utime = utime;
  planar_manipuland_desired.desired_state =
      this->EvalVectorInput(context, desired_brick_state_input_port_)
          ->CopyToVector();
  planar_manipuland_desired.desired_accel =
      this->EvalVectorInput(context, desired_brick_accel_input_port_)
          ->CopyToVector();

  // Sequentially serialize plant_state, finger_face_assignments and
  // planar_manipuland_desired into a message.
  std::vector<uint8_t> msg(plant_state.GetMessageSize() +
                           finger_face_assignments.GetMessageSize() +
                           planar_manipuland_desired.GetMessageSize());
  plant_state.Serialize(msg.data());
  int msg_start = plant_state.GetMessageSize();
  finger_face_assignments.Serialize(msg.data() + msg_start);
  msg_start += finger_face_assignments.GetMessageSize();
  planar_manipuland_desired.Serialize(msg.data() + msg_start);
  msg_start += planar_manipuland_desired.GetMessageSize();

  return msg;
}

systems::EventStatus SimToQPUdpPublisherSystem::PublishInputAsUdpMessage(
    const systems::Context<double>& context) const {
  const std::vector<uint8_t> output_msg = this->Serialize(context);
  struct sockaddr_in servaddr;
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(remote_port_);
  servaddr.sin_addr.s_addr = htonl(remote_address_);
  int status =
      sendto(file_descriptor_, output_msg.data(), output_msg.size(), 0,
             reinterpret_cast<struct sockaddr*>(&servaddr), sizeof(servaddr));
  if (status < 0) {
    throw std::runtime_error("Cannot send the UDP message.");
  }
  return systems::EventStatus::Succeeded();
}

QPControlUdpReceiverSystem::QPControlUdpReceiverSystem(
    int local_port, int num_fingers, multibody::BodyIndex brick_body_index)
    : num_fingers_{num_fingers},
      file_descriptor_{socket(AF_INET, SOCK_DGRAM, 0)},
      brick_body_index_{brick_body_index} {
  // The implementation of this class follows
  // https://www.cs.rutgers.edu/~pxk/417/notes/sockets/udp.html
  if (file_descriptor_ < 0) {
    throw std::runtime_error(
        " QPControlUdpReceiverSystem: cannot create a socket.");
  }
  struct sockaddr_in myaddr;
  myaddr.sin_family = AF_INET;
  // bind the socket to any valid IP address
  myaddr.sin_addr.s_addr = htonl(INADDR_ANY);
  myaddr.sin_port = htons(local_port);
  if (bind(file_descriptor_, reinterpret_cast<struct sockaddr*>(&myaddr),
           sizeof(myaddr)) < 0) {
    throw std::runtime_error(
        "QPControlUdpReceiverSystem: cannot bind the socket");
  }

  finger_control_state_index_ = this->DeclareAbstractState(
      std::make_unique<Value<std::unordered_map<
          Finger, multibody::ExternallyAppliedSpatialForce<double>>>>());
  brick_control_state_index_ = this->DeclareAbstractState(
      std::make_unique<Value<
          std::vector<multibody::ExternallyAppliedSpatialForce<double>>>>());

  this->DeclarePeriodicUnrestrictedUpdateEvent(
      kGripperUdpStatusPeriod, 0.,
      &QPControlUdpReceiverSystem::ProcessMessageAndStoreToAbstractState);

  qp_finger_control_output_port_ =
      this->DeclareAbstractOutputPort(
              "qp_fingers_control",
              &QPControlUdpReceiverSystem::OutputFingersControl)
          .get_index();
  qp_brick_control_output_port_ =
      this->DeclareAbstractOutputPort(
              "qp_brick_control",
              &QPControlUdpReceiverSystem::OutputBrickControl)
          .get_index();

  // Compute the number of bytes in each UDP packet.
  PlanarManipulandSpatialForces qp_finger_control_forces(num_fingers);
  PlanarManipulandSpatialForces qp_brick_control_forces(num_fingers);
  udp_message_size_ = qp_finger_control_forces.GetMessageSize() +
                      qp_brick_control_forces.GetMessageSize();
}

systems::EventStatus
QPControlUdpReceiverSystem::ProcessMessageAndStoreToAbstractState(
    const systems::Context<double>&, systems::State<double>* state) const {
  std::vector<uint8_t> buffer(udp_message_size_);
  struct sockaddr_in remaddr;
  socklen_t addrlen = sizeof(remaddr);
  const int recvlen =
      recvfrom(file_descriptor_, buffer.data(), buffer.size(), 0,
               reinterpret_cast<struct sockaddr*>(&remaddr), &addrlen);
  systems::AbstractValues& abstract_state = state->get_mutable_abstract_state();
  if (recvlen > 0) {
    // First deserialize the UDP message.
    PlanarManipulandSpatialForces qp_finger_control_udp(num_fingers_);
    int start = 0;
    qp_finger_control_udp.Deserialize(buffer.data());
    start += qp_finger_control_udp.GetMessageSize();
    PlanarManipulandSpatialForces qp_brick_control_udp(num_fingers_);
    qp_brick_control_udp.Deserialize(buffer.data() + start);
    start += qp_brick_control_udp.GetMessageSize();
    DRAKE_ASSERT(start == udp_message_size_);
    // Now convert PlanarManipulandSpatialForces struct to state.
    auto& finger_control_state =
        abstract_state.get_mutable_value(finger_control_state_index_)
            .get_mutable_value<std::unordered_map<
                Finger, multibody::ExternallyAppliedSpatialForce<double>>>();
    finger_control_state.clear();
    auto& brick_control_state =
        abstract_state.get_mutable_value(brick_control_state_index_)
            .get_mutable_value<std::vector<
                multibody::ExternallyAppliedSpatialForce<double>>>();
    brick_control_state.clear();
    for (int i = 0; i < num_fingers_; ++i) {
      if (qp_finger_control_udp.in_contact[i]) {
        const multibody::ExternallyAppliedSpatialForce<double>
            applied_spatial_force =
                qp_finger_control_udp.forces[i].ToSpatialForce(
                    brick_body_index_);
        finger_control_state.emplace(qp_finger_control_udp.forces[i].finger,
                                     applied_spatial_force);
      }
      if (qp_brick_control_udp.in_contact[i]) {
        brick_control_state.push_back(
            qp_brick_control_udp.forces[i].ToSpatialForce(brick_body_index_));
      }
    }
  }
  return systems::EventStatus::Succeeded();
}

void QPControlUdpReceiverSystem::OutputFingersControl(
    const systems::Context<double>& context,
    std::unordered_map<Finger,
                       multibody::ExternallyAppliedSpatialForce<double>>*
        fingers_control) const {
  fingers_control->clear();
  *fingers_control = context.get_abstract_state<std::unordered_map<
      Finger, multibody::ExternallyAppliedSpatialForce<double>>>(
      finger_control_state_index_);
}

void QPControlUdpReceiverSystem::OutputBrickControl(
    const systems::Context<double>& context,
    std::vector<multibody::ExternallyAppliedSpatialForce<double>>*
        brick_control) const {
  brick_control->clear();
  *brick_control = context.get_abstract_state<
      std::vector<multibody::ExternallyAppliedSpatialForce<double>>>(
      brick_control_state_index_);
}

QPtoSimUdpReceiverSystem::QPtoSimUdpReceiverSystem(int local_port,
                                                   int num_plant_states,
                                                   int num_fingers,
                                                   int num_brick_states,
                                                   int num_brick_accels)
    : file_descriptor_{socket(AF_INET, SOCK_DGRAM, 0)},
      num_plant_states_{num_plant_states},
      num_fingers_{num_fingers},
      num_brick_states_{num_brick_states},
      num_brick_accels_{num_brick_accels},
      message_count_{0} {
  // The implementation of this class follows
  // https://www.cs.rutgers.edu/~pxk/417/notes/sockets/udp.html
  if (file_descriptor_ < 0) {
    throw std::runtime_error(
        " QPControlUdpReceiverSystem: cannot create a socket.");
  }
  struct sockaddr_in myaddr;
  myaddr.sin_family = AF_INET;
  // bind the socket to any valid IP address
  myaddr.sin_addr.s_addr = htonl(INADDR_ANY);
  myaddr.sin_port = htons(local_port);
  if (bind(file_descriptor_, reinterpret_cast<struct sockaddr*>(&myaddr),
           sizeof(myaddr)) < 0) {
    throw std::runtime_error(
        "QPtoSimUdpReceiverSystem: cannot bind the socket");
  }

  plant_state_index_ = this->DeclareDiscreteState(num_plant_states_);
  finger_face_assignments_state_index_ = this->DeclareAbstractState(
      std::make_unique<Value<std::unordered_map<
          Finger, std::pair<BrickFace, Eigen::Vector2d>>>>());
  brick_state_index_ = this->DeclareDiscreteState(num_brick_states_);
  brick_accel_index_ = this->DeclareDiscreteState(num_brick_accels_);

  this->DeclarePeriodicUnrestrictedUpdateEvent(
      kGripperUdpStatusPeriod, 0., &QPtoSimUdpReceiverSystem::UpdateState);
  this->DeclareForcedUnrestrictedUpdateEvent(
      &QPtoSimUdpReceiverSystem::UpdateState);

  plant_state_output_port_ =
      this->DeclareVectorOutputPort(
              "qp_estimated_plant_state",
              systems::BasicVector<double>(num_plant_states_),
              &QPtoSimUdpReceiverSystem::OutputEstimatedPlantState)
          .get_index();
  finger_face_assignments_output_port_ =
      this->DeclareAbstractOutputPort(
              "qp_finger_face_assignments",
              &QPtoSimUdpReceiverSystem::OutputFingerFaceAssignments)
          .get_index();

  desired_brick_state_output_port_ =
      this->DeclareVectorOutputPort(
              "qp_desired_brick_state",
              systems::BasicVector<double>(num_brick_states_),
              &QPtoSimUdpReceiverSystem::OutputBrickDesiredState)
          .get_index();

  desired_brick_accel_output_port_ =
      this->DeclareVectorOutputPort(
              "qp_desired_brick_accel",
              systems::BasicVector<double>(num_brick_accels_),
              &QPtoSimUdpReceiverSystem::OutputBrickDesiredAccel)
          .get_index();

  // Compute the udp message size.
  PlanarPlantState plant_state_udp(num_plant_states_);
  FingerFaceAssignments finger_face_udp(num_fingers_);
  PlanarManipulandDesired manipuland_des_udp(num_brick_states_,
                                             num_brick_accels_);
  udp_message_size_ = plant_state_udp.GetMessageSize() +
                      finger_face_udp.GetMessageSize() +
                      manipuland_des_udp.GetMessageSize();
}

int QPtoSimUdpReceiverSystem::ReceiveUDPmsg(
    std::vector<uint8_t>* buffer) const {
  buffer->resize(udp_message_size_);
  struct sockaddr_in remaddr;
  socklen_t addrlen = sizeof(remaddr);
  const int recvlen =
      recvfrom(file_descriptor_, buffer->data(), buffer->size(), 0,
               reinterpret_cast<struct sockaddr*>(&remaddr), &addrlen);
  if (recvlen > 0) {
    message_count_++;
  }
  return recvlen;
}

systems::EventStatus QPtoSimUdpReceiverSystem::UpdateState(
    const systems::Context<double>&, systems::State<double>* state) const {
  // First deserialize the UDP message.
  std::vector<uint8_t> buffer;
  const int recvlen = ReceiveUDPmsg(&buffer);
  if (recvlen > 0) {
    PlanarPlantState plant_state_udp(num_plant_states_);
    int start = 0;
    plant_state_udp.Deserialize(buffer.data());
    start += plant_state_udp.GetMessageSize();
    FingerFaceAssignments finger_face_udp(num_fingers_);
    finger_face_udp.Deserialize(buffer.data() + start);
    start += finger_face_udp.GetMessageSize();
    PlanarManipulandDesired manipuland_des_udp(num_brick_states_,
                                               num_brick_accels_);
    manipuland_des_udp.Deserialize(buffer.data() + start);
    start += manipuland_des_udp.GetMessageSize();
    DRAKE_ASSERT(start == udp_message_size_);

    // Now convert the deserialized UDP message to state.
    Eigen::Ref<Eigen::VectorXd> plant_state =
        state->get_mutable_discrete_state(plant_state_index_)
            .get_mutable_value();
    plant_state = plant_state_udp.plant_state;

    auto& assignments = state->get_mutable_abstract_state<
        std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>>(
        finger_face_assignments_state_index_);
    assignments.clear();
    for (int i = 0; i < num_fingers_; ++i) {
      if (finger_face_udp.in_contact[i]) {
        assignments.emplace(
            finger_face_udp.finger_face_assignments[i].finger,
            std::make_pair(
                finger_face_udp.finger_face_assignments[i].brick_face,
                finger_face_udp.finger_face_assignments[i].p_BoBq_B));
      }
    }

    Eigen::Ref<Eigen::VectorXd> brick_state =
        state->get_mutable_discrete_state(brick_state_index_)
            .get_mutable_value();
    brick_state = manipuland_des_udp.desired_state;
    Eigen::Ref<Eigen::VectorXd> brick_accel =
        state->get_mutable_discrete_state(brick_accel_index_)
            .get_mutable_value();
    brick_accel = manipuland_des_udp.desired_accel;
  }
  return systems::EventStatus::Succeeded();
}

void QPtoSimUdpReceiverSystem::OutputEstimatedPlantState(
    const systems::Context<double>& context,
    systems::BasicVector<double>* plant_state) const {
  Eigen::VectorBlock<Eigen::VectorXd> output_vec =
      plant_state->get_mutable_value();
  output_vec = context.get_discrete_state(plant_state_index_).get_value();
}

void QPtoSimUdpReceiverSystem::OutputFingerFaceAssignments(
    const systems::Context<double>& context,
    std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>*
        finger_face_assignments) const {
  finger_face_assignments->clear();
  *finger_face_assignments = context.get_abstract_state<
      std::unordered_map<Finger, std::pair<BrickFace, Eigen::Vector2d>>>(
      finger_face_assignments_state_index_);
}

void QPtoSimUdpReceiverSystem::OutputBrickDesiredState(
    const systems::Context<double>& context,
    systems::BasicVector<double>* qp_desired_brick_state) const {
  Eigen::VectorBlock<Eigen::VectorXd> output_vec =
      qp_desired_brick_state->get_mutable_value();
  output_vec = context.get_discrete_state(brick_state_index_).get_value();
}

void QPtoSimUdpReceiverSystem::OutputBrickDesiredAccel(
    const systems::Context<double>& context,
    systems::BasicVector<double>* qp_desired_brick_accel) const {
  Eigen::VectorBlock<Eigen::VectorXd> output_vec =
      qp_desired_brick_accel->get_mutable_value();
  output_vec = context.get_discrete_state(brick_accel_index_).get_value();
}

QPControlUdpPublisherSystem::QPControlUdpPublisherSystem(
    double publish_period, int local_port, int remote_port,
    uint32_t remote_address, int num_fingers)
    : file_descriptor_{socket(AF_INET, SOCK_DGRAM, 0)},
      local_port_{local_port},
      remote_port_{remote_port},
      remote_address_{remote_address},
      num_fingers_{num_fingers} {
  struct sockaddr_in myaddr;
  memset(reinterpret_cast<char*>(&myaddr), 0, sizeof(myaddr));
  myaddr.sin_family = AF_INET;
  myaddr.sin_addr.s_addr = htonl(INADDR_ANY);
  myaddr.sin_port = htons(local_port_);
  int status =
      bind(file_descriptor_, reinterpret_cast<struct sockaddr*>(&myaddr),
           sizeof(myaddr));
  if (status < 0) {
    throw std::runtime_error(
        "QPControlUdpPublisherSystem: Cannot bind the UDP file descriptor.");
  }

  this->DeclareForcedPublishEvent(
      &QPControlUdpPublisherSystem::PublishInputAsUdpMessage);

  const double offset = 0.0;
  this->DeclarePeriodicPublishEvent(
      publish_period, offset,
      &QPControlUdpPublisherSystem::PublishInputAsUdpMessage);

  qp_fingers_control_input_port_ =
      this->DeclareAbstractInputPort(
              "qp_fingers_control",
              Value<std::unordered_map<
                  Finger, multibody::ExternallyAppliedSpatialForce<double>>>{})
          .get_index();
  qp_brick_control_input_port_ =
      this->DeclareAbstractInputPort(
              "qp_brick_control",
              Value<std::vector<
                  multibody::ExternallyAppliedSpatialForce<double>>>{})
          .get_index();
}

std::vector<uint8_t> QPControlUdpPublisherSystem::Serialize(
    const systems::Context<double>& context) const {
  // Construct PlanarPlantState
  const uint32_t utime = context.get_time() * 1e6;

  // Construct PlanarManipulandSpatialForces
  PlanarManipulandSpatialForces qp_fingers_control_udp(num_fingers_);
  qp_fingers_control_udp.utime = utime;
  const auto qp_fingers_control_input =
      this->get_input_port(qp_fingers_control_input_port_)
          .Eval<std::unordered_map<
              Finger, multibody::ExternallyAppliedSpatialForce<double>>>(
              context);
  int finger_control_index = 0;
  for (const auto& finger_control : qp_fingers_control_input) {
    qp_fingers_control_udp.in_contact[finger_control_index] = true;
    qp_fingers_control_udp.forces[finger_control_index].utime = utime;
    qp_fingers_control_udp.forces[finger_control_index].finger =
        finger_control.first;
    qp_fingers_control_udp.forces[finger_control_index].FromSpatialForce(
        finger_control.second);
    finger_control_index++;
  }
  PlanarManipulandSpatialForces qp_brick_control_udp(num_fingers_);
  qp_brick_control_udp.utime = utime;
  const auto qp_brick_control_input =
      this->get_input_port(qp_brick_control_input_port_)
          .Eval<std::vector<multibody::ExternallyAppliedSpatialForce<double>>>(
              context);
  int brick_control_index = 0;
  for (const auto brick_control : qp_brick_control_input) {
    qp_brick_control_udp.in_contact[brick_control_index] = true;
    qp_brick_control_udp.forces[brick_control_index].utime = utime;
    qp_brick_control_udp.forces[brick_control_index].FromSpatialForce(
        brick_control);
    brick_control_index++;
  }

  // Now serialize the UDP message
  std::vector<uint8_t> msg(qp_fingers_control_udp.GetMessageSize() +
                           qp_brick_control_udp.GetMessageSize());
  qp_fingers_control_udp.Serialize(msg.data());
  int msg_start = qp_fingers_control_udp.GetMessageSize();
  qp_brick_control_udp.Serialize(msg.data() + msg_start);
  msg_start += qp_brick_control_udp.GetMessageSize();

  return msg;
}

systems::EventStatus QPControlUdpPublisherSystem::PublishInputAsUdpMessage(
    const systems::Context<double>& context) const {
  const std::vector<uint8_t> output_msg = this->Serialize(context);
  struct sockaddr_in servaddr;
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(remote_port_);
  servaddr.sin_addr.s_addr = htonl(remote_address_);
  int status =
      sendto(file_descriptor_, output_msg.data(), output_msg.size(), 0,
             reinterpret_cast<struct sockaddr*>(&servaddr), sizeof(servaddr));
  if (status < 0) {
    throw std::runtime_error("Cannot send the UDP message.");
  }
  return systems::EventStatus::Succeeded();
}

PlanarGripperQPControllerUDP::PlanarGripperQPControllerUDP(
    int num_multibody_states, multibody::BodyIndex brick_index, int num_fingers,
    int num_brick_states, int num_brick_accels, int publisher_local_port,
    int publisher_remote_port, uint32_t publisher_remote_address,
    int receiver_local_port, double publish_period) {
  systems::DiagramBuilder<double> builder;
  // The UDP receiver receives from publisher of the remote QP controller, and
  // outputs signal to local simulation.
  auto qp_control_receiver = builder.AddSystem<QPControlUdpReceiverSystem>(
      receiver_local_port, num_fingers, brick_index);
  builder.ExportOutput(
      qp_control_receiver->get_qp_fingers_control_output_port(),
      "qp_fingers_control");
  builder.ExportOutput(qp_control_receiver->get_qp_brick_control_output_port(),
                       "qp_brick_control");

  // The UDP pulisher takes signal from local simulation, and publish them to
  // remote QP controller.
  auto sim_to_qp_publisher = builder.AddSystem<SimToQPUdpPublisherSystem>(
      publish_period, publisher_local_port, publisher_remote_port,
      publisher_remote_address, num_multibody_states, num_fingers,
      num_brick_states, num_brick_accels);
  builder.ExportInput(sim_to_qp_publisher->get_plant_state_input_port(),
                      "qp_estimated_plant_state");
  builder.ExportInput(
      sim_to_qp_publisher->get_finger_face_assignments_input_port(),
      "qp_finger_face_assignments");
  builder.ExportInput(sim_to_qp_publisher->get_desired_brick_state_input_port(),
                      "qp_desired_brick_state");
  builder.ExportInput(sim_to_qp_publisher->get_desired_brick_accel_input_port(),
                      "qp_desired_brick_accel");
  builder.BuildInto(this);
}

PlanarGripperSimulationUDP::PlanarGripperSimulationUDP(
    int num_multibody_states, int num_fingers, int num_brick_states,
    int num_brick_accels, int publisher_local_port, int publisher_remote_port,
    uint32_t publisher_remote_address, int receiver_local_port,
    double publish_period) {
  systems::DiagramBuilder<double> builder;

  qp_to_sim_receiver_ = builder.AddSystem<QPtoSimUdpReceiverSystem>(
      receiver_local_port, num_multibody_states, num_fingers, num_brick_states,
      num_brick_accels);
  builder.ExportOutput(
      qp_to_sim_receiver_->get_estimated_plant_state_output_port(),
      "qp_estimated_plant_state");
  builder.ExportOutput(
      qp_to_sim_receiver_->get_finger_face_assignments_output_port(),
      "qp_finger_face_assignments");
  builder.ExportOutput(
      qp_to_sim_receiver_->get_desired_brick_state_output_port(),
      "qp_desired_brick_state");
  builder.ExportOutput(
      qp_to_sim_receiver_->get_desired_brick_accel_output_port(),
      "qp_desired_brick_accel");

  auto qp_control_publisher = builder.AddSystem<QPControlUdpPublisherSystem>(
      publish_period, publisher_local_port, publisher_remote_port,
      publisher_remote_address, num_fingers);
  builder.ExportInput(qp_control_publisher->get_qp_fingers_control_input_port(),
                      "qp_fingers_control");
  builder.ExportInput(qp_control_publisher->get_qp_brick_control_input_port(),
                      "qp_brick_control");
  builder.BuildInto(this);
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
