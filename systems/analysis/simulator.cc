#include "drake/systems/analysis/simulator.h"

#include <chrono>
#include <limits>
#include <thread>

#include "drake/common/autodiff.h"
#include "drake/common/extract_double.h"

namespace drake {
namespace systems {

template <typename T>
void Simulator<T>::PauseIfTooFast() const {
  // If the user specifies to run as quickly as possible, then run at full
  // speed.
  if (target_realtime_rate_ <= 0) return;

  // If the actual real time rate is smaller than the target real time rate,
  // run as quickly as possible.
  if (real_time_rate_ < target_realtime_rate_ ) return;

  // Compute the desired sleep time (in seconds).
  const double sleep_time_s = weighted_virtual_time_ / target_realtime_rate_ -
      weighted_real_time_ * realtime_rate_weighting_factor_;

  std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(sleep_time_s * 1000000)));

/*
  // Determine the desired real time.
  const TimePoint desired_realtime = initial_realtime_ + Duration()

  const double simtime_now = ExtractDoubleOrThrow(get_context().get_time());
  const double simtime_passed = simtime_now - initial_simtime_;
  const TimePoint desired_realtime =
      initial_realtime_ + Duration(simtime_passed / target_realtime_rate_);
  // TODO(sherm1): Could add some slop to now() and not sleep if
  // we are already close enough. But what is a reasonable value?
  if (desired_realtime > Clock::now())
    std::this_thread::sleep_until(desired_realtime);
*/
}

template <typename T>
double Simulator<T>::get_actual_realtime_rate() const {
  /*
  const double simtime_now = ExtractDoubleOrThrow(get_context().get_time());
  const double simtime_passed = simtime_now - initial_simtime_;
  const Duration realtime_passed = Clock::now() - initial_realtime_;
  const double rate = (simtime_passed / realtime_passed.count());
  return rate;
   */
  return real_time_rate_;
}

template <class T>
void Simulator<T>::UpdateRealTimeRate() {
  const double virtual_time_now = ExtractDoubleOrThrow(
      get_context().get_time());
  const double virtual_time_passed = virtual_time_now - last_virtual_time_;
  last_virtual_time_ = virtual_time_now;
  const auto clock_now = Clock::now();
  const Duration real_time_passed = clock_now - last_real_time_;
  last_real_time_ = clock_now;
  (weighted_virtual_time_ *= realtime_rate_weighting_factor_) +=
      virtual_time_passed;
  (weighted_real_time_ *= realtime_rate_weighting_factor_) +=
      real_time_passed.count();
  real_time_rate_ = weighted_virtual_time_ / weighted_real_time_;
}

template <typename T>
void Simulator<T>::ResetStatistics() {
  integrator_->ResetStatistics();
  num_steps_taken_ = 0;
  num_discrete_updates_ = 0;
  num_unrestricted_updates_ = 0;
  num_publishes_ = 0;

  initial_simtime_ = ExtractDoubleOrThrow(get_context().get_time());
  initial_realtime_ = Clock::now();
}

template class Simulator<double>;
template class Simulator<AutoDiffXd>;

}  // namespace systems
}  // namespace drake
