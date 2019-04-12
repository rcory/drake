#pragma once

#include <memory>
#include <vector>

#include "drake/common/symbolic.h"
#include "drake/examples/pendulum/gen/pendulum_input.h"
#include "drake/examples/pendulum/gen/pendulum_params.h"
#include "drake/examples/pendulum/gen/pendulum_state.h"
#include "drake/geometry/scene_graph.h"
#include "drake/systems/framework/basic_vector.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace examples {
namespace pendulum {

/// A model of a simple pendulum
/// @f[ ml^2 \ddot\theta + b\dot\theta + mgl\sin\theta = \tau @f]
///
/// @system{PendulumPlant,
///    @input_port{tau},
///    @output_port{state} @output_port{geometry_pose}
/// }
///
/// @tparam T The vector element type, which must be a valid Eigen scalar.
///
/// Instantiated templates for the following kinds of T's are provided:
///
/// - double
/// - AutoDiffXd
/// - symbolic::Expression
template <typename T>
class PendulumPlant : public systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(PendulumPlant);

  /** Constructs a default plant. */
  PendulumPlant();

  /// Scalar-converting copy constructor.  See @ref system_scalar_conversion.
  template <typename U>
  explicit PendulumPlant(const PendulumPlant<U>&);

  ~PendulumPlant() override;

  /// Returns the input port to the externally applied force.
  const systems::InputPort<T>& get_actuation_input_port() const;

  /// Returns the port to output state.
  const systems::OutputPort<T>& get_continuous_state_output_port() const;

  geometry::SourceId source_id() const { return source_id_; }
  geometry::FrameId frame_id() const { return frame_id_; }

  /// Registers this system as a source for the SceneGraph, adds the
  /// pendulum geometry, and creates the geometry_pose_output_port for this
  /// system.  This must be called before the a SceneGraph's Context is
  /// allocated.  It can only be called once.
  // TODO(russt): this call only accepts doubles (not T) until SceneGraph
  // supports symbolic.
  void RegisterGeometry(const PendulumParams<double>& params,
                        geometry::SceneGraph<double>* scene_graph);

  /// Returns the port to output the pose to SceneGraph.  Users must call
  /// RegisterGeometry() first to enable this port.
  const systems::OutputPort<T>& get_geometry_poses_output_port() const;

  const systems::InputPort<T>& get_geometry_query_input_port() const;

  /// There is no direct-feedthrough in this system.
  optional<bool> DoHasDirectFeedthrough(int, int) const override {
    return false;
  }

  /// Calculates the kinetic + potential energy.
  T CalcTotalEnergy(const systems::Context<T>& context) const;

  /// Evaluates the input port and returns the scalar value
  /// of the commanded torque.
  T get_tau(const systems::Context<T>& context) const {
    return this->get_actuation_input_port().Eval(context)(0);
  }

  static const PendulumState<T>& get_continuous_state(
      const systems::ContinuousState<T>& cstate) {
    return dynamic_cast<const PendulumState<T>&>(cstate.get_vector());
  }

  static const PendulumState<T>& get_continuous_state(
      const systems::Context<T>& context) {
    return get_continuous_state(context.get_continuous_state());
  }

  static PendulumState<T>& get_mutable_continuous_state(
      systems::ContinuousState<T>* cstate) {
    return dynamic_cast<PendulumState<T>&>(cstate->get_mutable_vector());
  }

  static PendulumState<T>& get_mutable_continuous_state(
      systems::Context<T>* context) {
    return get_mutable_continuous_state(
        &context->get_mutable_continuous_state());
  }

  const PendulumParams<T>& get_parameters(
      const systems::Context<T>& context) const {
    return this->template GetNumericParameter<PendulumParams>(context, 0);
  }

  PendulumParams<T>& get_mutable_parameters(
      systems::Context<T>* context) const {
    return this->template GetMutableNumericParameter<PendulumParams>(
        context, 0);
  }

 protected:
  // Constructor that specifies scalar-type conversion support.
  // @param converter scalar-type conversion support helper (i.e., AutoDiff,
  // etc.); pass a default-constructed object if such support is not desired.
  explicit PendulumPlant(systems::SystemScalarConverter converter)
  : systems::LeafSystem<T>(std::move(converter)) {
    this->DeclareVectorInputPort(PendulumInput<T>());
    state_port_ = this->DeclareVectorOutputPort(PendulumState<T>(),
                                                &PendulumPlant::CopyStateOut)
                      .get_index();

    this->DeclareContinuousState(PendulumState<T>(), 1 /* num_q */,
                                 1 /* num_v */, 0 /* num_z */);
    this->DeclareNumericParameter(PendulumParams<T>());
  }

  // Computes the actual physics.
  void dynamics(const PendulumState<T>& state, const PendulumParams<T>& params,
                T tau, PendulumState<T>* derivative_vector) const;

 private:
  systems::OutputPortIndex AllocateGeometryPoseOutputPort();

  systems::InputPortIndex AllocateGeometryQueryInputPort();

  // This is the calculator method for the state output port.
  void CopyStateOut(const systems::Context<T>& context,
                    PendulumState<T>* output) const;

  // Calculator method for the pose output port.
  void CopyPoseOut(const systems::Context<T>& context,
                   geometry::FramePoseVector<T>* poses) const;

  void DoCalcTimeDerivatives(
      const systems::Context<T>& context,
      systems::ContinuousState<T>* derivatives) const override;

  // Port handles.
  int state_port_{-1};
  int geometry_pose_port_{-1};
  int geometry_query_port_{-1};

  // Geometry source identifier for this system to interact with SceneGraph.
  geometry::SourceId source_id_{};
  // The id for the pendulum (arm + point mass) frame.
  geometry::FrameId frame_id_{};
};

}  // namespace pendulum
}  // namespace examples
}  // namespace drake
