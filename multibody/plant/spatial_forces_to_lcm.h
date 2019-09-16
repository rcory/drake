#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/drake_deprecated.h"
#include "drake/lcmt_spatial_forces_for_viz.hpp"
#include "drake/multibody/plant/contact_results.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"

namespace drake {
namespace multibody {

/** A System that encodes std::vector<SpatialForceOutput> into a
 lcmt_spatial_forces_for_viz message. It has a single input port with type
 std::vector<SpatialForceOutput> and a single
 output port with lcmt_spatial_forces_for_viz.

 @tparam T The scalar type. Must be a valid Eigen scalar.

 Instantiated templates for the following kinds of T's are provided:

 - double
 - AutoDiffXd

 They are already available to link against in the containing library. No other
 values for T are currently supported.
 */
template <typename T>
class SpatialForcesToLcmSystem final : public systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SpatialForcesToLcmSystem)

  /** Constructs a SpatialForcesToLcmSystem.
   @param plant The MultibodyPlant that the std::vector<SpatialForcesOutput> is
          generated from.
   @pre The `plant` must be finalized already. The input port of this system
        must be connected to the corresponding output port of `plant`
        (either directly or from an exported port in a Diagram).
  */
  explicit SpatialForcesToLcmSystem(const MultibodyPlant<T>& plant);

  /** Scalar-converting copy constructor.  */
  template <typename U>
  explicit SpatialForcesToLcmSystem(const SpatialForcesToLcmSystem<U>& other)
      : systems::LeafSystem<T>(), body_names_(other.body_names_) {}

  const systems::InputPort<T>& get_spatial_forces_input_port() const;
  const systems::OutputPort<T>& get_lcm_message_output_port() const;

 private:
  // Allow different specializations to access each other's private data for
  // scalar conversion.
  template <typename U> friend class SpatialForcesToLcmSystem;

  void CalcLcmSpatialForcesOutput(const systems::Context<T>& context,
      lcmt_spatial_forces_for_viz* output) const;

  // Named indices for the i/o ports.
  static constexpr int spatial_forces_input_port_index_{0};
  static constexpr int message_output_port_index_{0};

  // A mapping from body index values to body names.
  std::vector<std::string> body_names_;
};

/** Implements ConnectSpatialForcesToDrakeVisualizer, but using @p
 contact_results_port to explicitly specify the output port used to get
 contact results for @p multibody_plant.  This is required, for instance,
 when the MultibodyPlant is inside a Diagram, and the Diagram exports the
 pose bundle port.

 @pre contact_results_port must be connected to the contact_results_port of
 @p multibody_plant.

 @see ConnectSpatialForcesToDrakeVisualizer().

 @ingroup visualization
 */
systems::lcm::LcmPublisherSystem* ConnectSpatialForcesToDrakeVisualizer(
    systems::DiagramBuilder<double>* builder,
    const MultibodyPlant<double>& multibody_plant,
    const systems::OutputPort<double>& spatial_forces_output_port,
    lcm::DrakeLcmInterface* lcm = nullptr);

}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class drake::multibody::SpatialForcesToLcmSystem)
