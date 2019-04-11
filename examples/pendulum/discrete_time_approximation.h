#pragma once

#include "drake/examples/pendulum/pendulum_plant.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/examples/pendulum/gen/pendulum_input.h"
#include "drake/examples/pendulum/gen/pendulum_params.h"
#include "drake/examples/pendulum/gen/pendulum_state.h"

namespace drake {
namespace examples {
namespace pendulum {

template <typename T>
class DiscretePendulumPlant final : public PendulumPlant<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DiscretePendulumPlant);

  /** Constructs a default plant */
  DiscretePendulumPlant(double time_step = 0)
      : PendulumPlant<T>(systems::SystemTypeTag<DiscretePendulumPlant>{}),
        time_step_(time_step) {
    if (time_step_ == 0) {
      throw std::logic_error(
          "time_step must be greater than zero. For continuous time "
          "pendulum plant use the PendulumPlant system directly.");
    }
    // Ensure time step is positive.
    DRAKE_DEMAND(time_step_ > 0);

    // Ensure the underlying PendulumPlant only has continuous states.
    DRAKE_DEMAND(this->num_discrete_state_groups() == 0);

    // Declare the discrete state held by this class.
    this->DeclareDiscreteState(PendulumState<T>());
    this->DeclarePeriodicDiscreteUpdateEvent(
        time_step_, 0, &DiscretePendulumPlant::DoDiscreteStateUpdate);
    this->DeclareForcedDiscreteUpdateEvent(
        &DiscretePendulumPlant::DoDiscreteStateUpdate);
  }

  /// Scalar-converting copy constructor.  See @ref system_scalar_conversion.
  template <typename U>
  explicit DiscretePendulumPlant(const DiscretePendulumPlant<U>& p)
      : DiscretePendulumPlant<T>(p.time_step()) {}

  ~DiscretePendulumPlant() override {}

  /* The period and forced discrete state update handler */
  systems::EventStatus DoDiscreteStateUpdate(
      const systems::Context<T>& context,
      systems::DiscreteValues<T>* discrete_state) const {

    // Get the current discrete state.
    const PendulumState<T>& state = get_discrete_state(context);
    PendulumState<T>& mutable_state =
        get_mutable_discrete_state(discrete_state);
    const PendulumParams<T>& params = this->get_parameters(context);

    // Compute the state derivatives.
    PendulumState<T> derivatives;
    this->dynamics(mutable_state, params, this->get_tau(context), &derivatives);

    // Update the discrete state using Explicit Euler.
    mutable_state.set_theta(state.theta() + derivatives.theta() * time_step_);
    mutable_state.set_thetadot(state.thetadot() +
                               derivatives.thetadot() * time_step_);

    return systems::EventStatus::Succeeded();
  }

  static const PendulumState<T>& get_discrete_state(
      const systems::DiscreteValues<T>& dstate) {
    return dynamic_cast<const PendulumState<T>&>(dstate.get_vector());
  }

  static const PendulumState<T>& get_discrete_state(
      const systems::Context<T>& context) {
    return get_discrete_state(context.get_discrete_state());
  }

  static PendulumState<T>& get_mutable_discrete_state(
      systems::DiscreteValues<T>* dstate) {
    return dynamic_cast<PendulumState<T>&>(dstate->get_mutable_vector());
  }

  static PendulumState<T>& get_mutable_discrete_state(
      systems::Context<T>* context) {
    return get_mutable_discrete_state(&context->get_mutable_discrete_state());
  }

  double time_step() const {return time_step_; }

 private:
  // If the plant is modeled as a discrete system with periodic updates,
  // time_step_ corresponds to the period of those updates. Otherwise, if the
  // plant is modeled as a continuous system, it is exactly zero.
  double time_step_{0};
};

}
}
}

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::examples::pendulum::DiscretePendulumPlant)
