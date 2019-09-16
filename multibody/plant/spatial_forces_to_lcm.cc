#include "drake/multibody/plant/spatial_forces_to_lcm.h"

#include <memory>

#include "drake/common/value.h"
#include "drake/lcmt_spatial_forces_for_viz.hpp"
#include "drake/multibody/plant/spatial_force_output.h"

namespace drake {
namespace multibody {

using systems::Context;

template <typename T>
SpatialForcesToLcmSystem<T>::SpatialForcesToLcmSystem(
    const MultibodyPlant<T>& plant)
    : systems::LeafSystem<T>() {
  DRAKE_DEMAND(plant.is_finalized());
  const int body_count = plant.num_bodies();

  body_names_.reserve(body_count);
  using std::to_string;
  for (BodyIndex i{0}; i < body_count; ++i) {
    const Body<T>& body = plant.get_body(i);
    body_names_.push_back(body.name() + "(" + to_string(body.model_instance()) +
                          ")");
  }
  this->set_name("SpatialForcesToLcmSystem");
  // Must be the first declared input port to be compatible with the constexpr
  // declaration of spatial_forces_input_port_index_.
  this->DeclareAbstractInputPort(Value<std::vector<SpatialForceOutput<T>>>());
  this->DeclareAbstractOutputPort(
      &SpatialForcesToLcmSystem::CalcLcmSpatialForcesOutput);
}

template <typename T>
const systems::InputPort<T>&
SpatialForcesToLcmSystem<T>::get_spatial_forces_input_port() const {
  return this->get_input_port(spatial_forces_input_port_index_);
}

template <typename T>
const systems::OutputPort<T>&
SpatialForcesToLcmSystem<T>::get_lcm_message_output_port() const {
  return this->get_output_port(message_output_port_index_);
}

template <typename T>
void SpatialForcesToLcmSystem<T>::CalcLcmSpatialForcesOutput(
    const Context<T>& context, lcmt_spatial_forces_for_viz* output) const {
  // Get input / output.
  const auto& spatial_forces =
      this->EvalAbstractInput(context, spatial_forces_input_port_index_)
          ->template get_value<std::vector<SpatialForceOutput<T>>>();
  auto& msg = *output;

  // Time in microseconds.
  msg.timestamp = static_cast<int64_t>(
      ExtractDoubleOrThrow(context.get_time()) * 1e6);
  msg.num_forces_to_visualize = static_cast<int>(spatial_forces.size());
  msg.forces.resize(msg.num_forces_to_visualize);

  for (int i = 0; i < msg.num_forces_to_visualize; ++i) {
    lcmt_spatial_force_for_viz& force_msg = msg.forces[i];
    const SpatialForceOutput<T>& force_output = spatial_forces[i];
    force_msg.timestamp = msg.timestamp;

    auto write_double3 = [](const Vector3<T>& src, double* dest) {
      dest[0] = ExtractDoubleOrThrow(src(0));
      dest[1] = ExtractDoubleOrThrow(src(1));
      dest[2] = ExtractDoubleOrThrow(src(2));
    };
    write_double3(force_output.p_W, force_msg.p_W);
    write_double3(force_output.F_p_W.translational(), force_msg.force_W);
    write_double3(force_output.F_p_W.rotational(), force_msg.torque_W);
  }
}

systems::lcm::LcmPublisherSystem* ConnectSpatialForcesToDrakeVisualizer(
    systems::DiagramBuilder<double>* builder,
    const MultibodyPlant<double>& multibody_plant,
    lcm::DrakeLcmInterface* lcm) {
  return ConnectSpatialForcesToDrakeVisualizer(
      builder, multibody_plant,
      multibody_plant.get_spatial_forces_output_port(), lcm);
}

systems::lcm::LcmPublisherSystem* ConnectSpatialForcesToDrakeVisualizer(
    systems::DiagramBuilder<double>* builder,
    const MultibodyPlant<double>& multibody_plant,
    const systems::OutputPort<double>& spatial_forces_port,
    lcm::DrakeLcmInterface* lcm) {
  DRAKE_DEMAND(builder != nullptr);

  auto spatial_forces_to_lcm =
      builder->template AddSystem<SpatialForcesToLcmSystem<double>>(
          multibody_plant);
  spatial_forces_to_lcm->set_name("spatial_forces_to_lcm");

  auto spatial_forces_publisher = builder->AddSystem(
      systems::lcm::LcmPublisherSystem::Make<lcmt_spatial_forces_for_viz>(
          "SPATIAL_FORCES", lcm, 1.0 / 60 /* publish period */));
  spatial_forces_publisher->set_name("spatial_forces_publisher");

  builder->Connect(
      spatial_forces_port, spatial_forces_to_lcm->get_input_port(0));
  builder->Connect(spatial_forces_to_lcm->get_output_port(0),
                   spatial_forces_publisher->get_input_port());

  return spatial_forces_publisher;
}

}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class drake::multibody::SpatialForcesToLcmSystem)
