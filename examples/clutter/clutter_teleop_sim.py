import argparse
import numpy as np
import os

from pydrake.geometry import MeshcatVisualizer, StartMeshcat
from pydrake.multibody.plant import (
    MultibodyPlant,
    AddMultibodyPlant,
    MultibodyPlantConfig,
)
from pydrake.multibody.parsing import (
    ProcessModelDirectives,
    LoadModelDirectives,
    Parser,
)
from pydrake.systems.analysis import (
    ApplySimulatorConfig,
    Simulator,
    SimulatorConfig,
)
from pydrake.systems.controllers import PidController
from pydrake.systems.framework import (
    LeafSystem,
    DiagramBuilder,
)
from pydrake.systems.primitives import ConstantVectorSource
from pydrake.systems.primitives import Multiplexer
from pydrake.visualization import (
    ApplyVisualizationConfig,
    VisualizationConfig,
)

# Define the gamepad button to motion command mappings.
Command = {
    "N_YAW": 1,  # negative yaw
    "P_YAW": 2,  # positive yaw
    "N_X": 15,  # negative x
    "P_X": 14,  # positive x
    "N_Y": 12,  # negative y
    "P_Y": 13,  # positive y
}


class GamepadControl(LeafSystem):
    def __init__(self, meshcat, plant):
        """
        A Drake LeafSystem that monitors the gamepad button presses via the Meshcat
        visualizer, and keeps an internal discrete state of the paddle's commanded
        positions. It has no input ports, an declares a single vector valued output
        port that contains the current commanded positions of the paddle.
        """
        LeafSystem.__init__(self)

        paddle_model_index = plant.GetModelInstanceByName("paddle")

        # Declare the vector valued output port of commanded positions.
        assert(plant.num_positions(paddle_model_index) == 3)
        self.DeclareVectorOutputPort(
            "paddle_position",
            plant.num_positions(paddle_model_index),
            self.OutputPaddlePosition)

        # Declare a discrete state that stores the current commanded positions.
        self._paddle_state_index = self.DeclareDiscreteState(
            plant.num_positions(paddle_model_index))  # paddle position

        # Declare an update period for this system, and the update callback
        # function.
        self._time_step = 0.05
        self.DeclarePeriodicDiscreteUpdateEvent(
            self._time_step, 0, self.UpdatePaddlePosition
        )

        # Store a reference to the Meshcat visualizer (for gamepad interface) as well as
        # a reference to the MultibodyPlant.
        self._meshcat = meshcat
        self._plant = plant
        self._paddle_model_index = paddle_model_index

        # Callback to set the initial value of the discrete state.
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)

    def Initialize(self, context, discrete_state):
        """Sets values to zero (default pose).
        """
        zero_vector = np.zeros(self._plant.num_positions(self._paddle_model_index))
        discrete_state.set_value(self._paddle_state_index, zero_vector)

    def UpdatePaddlePosition(self, context, output):
        """
        Queries the gamepad button presses from the Meshcat visualizer, and sets the commanded
        positions of the paddle.
        """
        x_y_theta_offset = [0, 0, 0]
        gamepad = self._meshcat.GetGamepad()

        if (gamepad.index is not None):
            motion_scale_factor = 1e-2
            # Set the x-y motion
            x_y_theta_offset[0] = (gamepad.button_values[Command["P_X"]] -
                                   gamepad.button_values[Command["N_X"]]) * motion_scale_factor
            x_y_theta_offset[1] = (gamepad.button_values[Command["P_Y"]] -
                                   gamepad.button_values[Command["N_Y"]]) * motion_scale_factor
            x_y_theta_offset[2] = (gamepad.button_values[Command["P_YAW"]] -
                                   gamepad.button_values[Command["N_YAW"]]) * motion_scale_factor * 10

        x_y_theta_current = output.get_value(self._paddle_state_index)
        x_y_theta_new = x_y_theta_current + x_y_theta_offset

        # Check against joint limits.
        plant_joint_pos_low_limits = self._plant.GetPositionLowerLimits()
        plant_joint_pos_up_limits = self._plant.GetPositionUpperLimits()

        # Extract the limits for the paddle only (joints are ordered x, y, theta).
        paddle_joint_pos_low_limits = self._plant.GetPositionsFromArray(
            self._paddle_model_index, plant_joint_pos_low_limits)
        paddle_joint_pos_up_limits = self._plant.GetPositionsFromArray(
            self._paddle_model_index, plant_joint_pos_up_limits)

        # Clip the commanded positions.
        x_y_theta_limited = np.clip(
            x_y_theta_new,
            paddle_joint_pos_low_limits,
            paddle_joint_pos_up_limits)

        output.set_value(self._paddle_state_index, x_y_theta_limited)

    def OutputPaddlePosition(self, context, output):
        """Actually sets the output port of this system to hold the current commanded positions.
        """
        output.set_value(
            context.get_discrete_state(
                self._paddle_state_index).value())


def run(*, local_dir):
    """Runs a simulation from the given model directives.
    """
    if not local_dir:
        local_dir = os.path.dirname(__file__)

    builder = DiagramBuilder()

    # Plant configuration (timestep and contact parameters).
    plant_config = MultibodyPlantConfig()
    plant_config.time_step = 11e-3
    plant_config.discrete_contact_solver = "sap"

    # Create the multibody plant and scene graph.
    sim_plant, scene_graph = AddMultibodyPlant(
        config=plant_config,
        builder=builder)

    # Load simulation model directives.
    parser = Parser(sim_plant)
    directives = LoadModelDirectives(local_dir + "/sim_directives.dmd.yaml")

    # Add a custom package map to the local repository location.
    package_map = parser.package_map()
    package_map.Add("clutter", local_dir)

    ProcessModelDirectives(
        directives=directives,
        plant=sim_plant,
        parser=parser)

    # Now the simulation plant is complete.
    sim_plant.Finalize()

    paddle_model_index = sim_plant.GetModelInstanceByName("paddle")

    # Create the Meshcat visualizer.
    meshcat = StartMeshcat()

    # Add and connect the paddle's controller.
    paddle_pid_controller = builder.AddSystem(
        PidController(kp=[20, 20, 0.5], ki=[0, 0, 0], kd=[2.5, 2.5, 0.075]))

    # Add gamepad teleop to the builder, and connect to the controller.
    gamepad_control = builder.AddSystem(GamepadControl(meshcat, sim_plant))
    mux = builder.AddSystem(Multiplexer([sim_plant.num_positions(
        paddle_model_index), sim_plant.num_velocities(paddle_model_index)]))
    zero_vel_src = builder.AddSystem(ConstantVectorSource([0, 0, 0]))
    builder.Connect(gamepad_control.get_output_port(), mux.get_input_port(0))
    builder.Connect(zero_vel_src.get_output_port(), mux.get_input_port(1))
    builder.Connect(
        mux.get_output_port(),
        paddle_pid_controller.get_input_port_desired_state())

    sim_paddle_instance = sim_plant.GetModelInstanceByName("paddle")
    builder.Connect(
        sim_plant.get_state_output_port(sim_paddle_instance),
        paddle_pid_controller.get_input_port_estimated_state())
    builder.Connect(
        paddle_pid_controller.get_output_port_control(),
        sim_plant.get_actuation_input_port())

    # Set visualization configuration.
    visualization_config = VisualizationConfig()
    visualization_config.publish_contacts = True
    visualization_config.publish_proximity = False
    visualization_config.enable_alpha_sliders = True
    visualization_config.publish_period = 0.05

    # Add visualization.
    ApplyVisualizationConfig(visualization_config, builder, meshcat=meshcat)

    # Simulator configuration (integrator and publisher parameters).
    simulator_config = SimulatorConfig(
        target_realtime_rate=1)

    # Build the diagram and its simulator.
    diagram = builder.Build()
    simulator = Simulator(diagram)
    ApplySimulatorConfig(simulator_config, simulator)

    meshcat.AddButton("Stop Simulation", "Escape")
    print("To stop simulation, press `Stop Simulation` in the Meshcat control panel or")
    print("press keyboard `Escape` (with Meshcat window in focus)")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.5)
    meshcat.DeleteButton("Stop Simulation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--working_directory", required=False,
        help="Full path to the local directory for directives file and models folder.")
    args = parser.parse_args()
    run(local_dir=args.working_directory)


if __name__ == "__main__":
    main()
