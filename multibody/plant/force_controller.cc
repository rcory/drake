#include "drake/multibody/plant/force_controller.h"

#include <memory>

#include "drake/lcmt_contact_results_for_viz.hpp"

namespace drake {
namespace multibody {

using systems::Context;

template <typename T>
ForceController<T>::ForceController(
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
  this->set_name("ForceController");
  // Must be the first declared input port to be compatible with the constexpr
  // declaration of contact_result_input_port_index_.
  this->DeclareAbstractInputPort(Value<ContactResults<T>>());
  this->DeclareAbstractOutputPort(
      &ForceController::CalcLcmContactOutput);
  this->DeclareVectorOutputPort("tau", systems::BasicVector<T>(7),
                                &ForceController::calc_force);
}

template <typename T>
void ForceController<T>::calc_force(
    const systems::Context<T>& context,
    systems::BasicVector<T>* output_vector) const {

  const auto& contact_results = get_contact_result_input_port().
      template Eval<ContactResults<T>>(context);

  VectorX<T> const_vec(7);
  const_vec.setConstant(contact_results.num_contacts());
  output_vector->SetFromVector(const_vec);
}

template <typename T>
const systems::InputPort<T>&
ForceController<T>::get_contact_result_input_port() const {
  return this->get_input_port(contact_result_input_port_index_);
}

template <typename T>
const systems::OutputPort<T>&
ForceController<T>::get_lcm_message_output_port() const {
  return this->get_output_port(message_output_port_index_);
}

template <typename T>
void ForceController<T>::CalcLcmContactOutput(
    const Context<T>& context, lcmt_contact_results_for_viz* output) const {
  // Get input / output.
  const auto& contact_results = get_contact_result_input_port().
      template Eval<ContactResults<T>>(context);
  auto& msg = *output;

  // Time in microseconds.
  msg.timestamp = static_cast<int64_t>(
      ExtractDoubleOrThrow(context.get_time()) * 1e6);
  msg.num_contacts = contact_results.num_contacts();
  msg.contact_info.resize(msg.num_contacts);

  for (int i = 0; i < contact_results.num_contacts(); ++i) {
    lcmt_contact_info_for_viz& info_msg = msg.contact_info[i];
    info_msg.timestamp = msg.timestamp;

    const PointPairContactInfo<T>& contact_info =
        contact_results.point_pair_contact_info(i);

    info_msg.body1_name = body_names_.at(contact_info.bodyA_index());
    info_msg.body2_name = body_names_.at(contact_info.bodyB_index());

    auto write_double3 = [](const Vector3<T>& src, double* dest) {
      dest[0] = ExtractDoubleOrThrow(src(0));
      dest[1] = ExtractDoubleOrThrow(src(1));
      dest[2] = ExtractDoubleOrThrow(src(2));
    };
    write_double3(contact_info.contact_point(), info_msg.contact_point);
    write_double3(contact_info.contact_force(), info_msg.contact_force);
    write_double3(contact_info.point_pair().nhat_BA_W, info_msg.normal);
  }
}

}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class drake::multibody::ForceController)
