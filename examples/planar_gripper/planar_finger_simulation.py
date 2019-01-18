"""Provides an example of a 2d gripper, with prismatic fingertips."""

import argparse

from pydrake.common import FindResourceOrThrow
from pydrake.geometry import (ConnectDrakeVisualizer, SceneGraph)
from pydrake.lcm import DrakeLcm
from pydrake.multibody.tree import UniformGravityFieldElement, JointActuatorIndex, JointIndex
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import DiagramBuilder, LeafSystem, BasicVector
from pydrake.systems.analysis import Simulator
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.plant import (
    MultibodyPlant,
    ConnectContactResultsToDrakeVisualizer
)
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.primitives import (
    Sine,
    Adder,
    ConstantVectorSource,
    Multiplexer
)
import numpy as np


def weld_gripper_frames(plant):
    outer_radius = 0.19  # 19 - 22
    f1_angle = 0*60*(np.pi/180.)

    XT = RigidTransform(RollPitchYaw([0, 0, 0]), [0, 0, outer_radius])

    # Weld the first finger
    # X_PC1 = RigidTransform(RollPitchYaw([0, 0, 0]), [0, 0, outer_radius])
    X_PC1 = RigidTransform(RollPitchYaw([f1_angle, 0, 0]), [0, 0, 0]).multiply(XT)
    child_frame = plant.GetFrameByName("finger_base")
    plant.WeldFrames(plant.world_frame(),
                       child_frame, X_PC1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target_realtime_rate", type=float, default=1.0,
        help="Desired rate relative to real time.  See documentation for "
             "Simulator::set_target_realtime_rate() for details.")
    parser.add_argument(
        "--simulation_time", type=float, default=10.0,
        help="Desired duration of the simulation in seconds.")
    parser.add_argument(
        "--time_step", type=float, default=5e-4,
        help="If greater than zero, the plant is modeled as a system with "
             "discrete updates and period equal to this time_step. "
             "If 0, the plant is modeled as a continuous system.")

    args = parser.parse_args()

    file_name = FindResourceOrThrow(
        "drake/examples/planar_gripper/planar_finger.sdf")
    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    plant = builder.AddSystem(MultibodyPlant(time_step=args.time_step))
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    plant_id = Parser(plant=plant).AddModelFromFile(file_name, "finger")
    weld_gripper_frames(plant)

    # Adds the object to be manipulated.
    object_file_name = FindResourceOrThrow(
        "drake/examples/planar_gripper/1dof_brick.sdf")
    object_id = Parser(plant=plant).AddModelFromFile(object_file_name, "object")

    # Create the controlled plant. Contains only the finger (no objects).
    control_plant = MultibodyPlant(time_step=args.time_step)
    control_plant_id = \
        Parser(plant=control_plant).AddModelFromFile(file_name, "finger")
    weld_gripper_frames(control_plant)

    # Finalize the plants.
    plant.Finalize()
    control_plant.Finalize()
    assert plant.geometry_source_is_registered()

# ===== Inverse Dynamics Source ============================================
    Kp = np.array([1, 1]) * 1500
    Kd = np.array([1, 1]) * 500
    Ki = np.array([1, 1]) * 500
    id_controller = builder.AddSystem(
        InverseDynamicsController(control_plant, Kp, Ki, Kd, False))

    # Connect the ID controller
    builder.Connect(plant.get_state_output_port(plant_id),
                    id_controller.get_input_port_estimated_state())

    # Sine reference
    amplitudes = [0.2, 0.7]
    frequencies = [3, 6.1]
    phases = [0, 0.2]
    sine_source = builder.AddSystem(Sine(amplitudes, frequencies, phases))
    smux = builder.AddSystem(Multiplexer([2, 2]))  # [q, qdot]

    # Add Sine offsets
    adder = builder.AddSystem(Adder(2, 2))
    offsets = builder.AddSystem(
        ConstantVectorSource([-0.65, 1.2]))
    builder.Connect(sine_source.get_output_port(0),
                    adder.get_input_port(0))
    builder.Connect(offsets.get_output_port(0),
                    adder.get_input_port(1))

    # Connect the offset Sine reference to the IDC reference input
    builder.Connect(adder.get_output_port(0), smux.get_input_port(0))
    builder.Connect(sine_source.get_output_port(1), smux.get_input_port(1))
    builder.Connect(smux.get_output_port(0),
                    id_controller.get_input_port_desired_state())

    builder.Connect(id_controller.get_output_port_control(),
                    plant.get_actuation_input_port())

    # Connect the SceneGraph with MBP.
    builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port())
    builder.Connect(
        plant.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()))

    ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)

    # Publish contact results
    lcm = DrakeLcm()
    ConnectContactResultsToDrakeVisualizer(builder, plant, lcm)

    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(
        plant, diagram_context)

    # Set the plant initial conditions.
    sh_pin = plant.GetJointByName("finger_ShoulderJoint")
    el_pin = plant.GetJointByName("finger_ElbowJoint")
    sh_pin.set_angle(context=plant_context, angle=-0.65)
    el_pin.set_angle(context=plant_context, angle=1.2)

    # Set the box initial conditions
    box_pin = plant.GetJointByName("box_pin_joint")
    box_pin.set_angle(context=plant_context, angle=0)

    # Start the simulator
    simulator = Simulator(diagram, diagram_context)
    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(args.target_realtime_rate)
    simulator.Initialize()

    # simulator.StepTo(args.simulation_time)  # <-- can't cntrl-c this version
    while simulator.get_context().get_time() < args.simulation_time:
        simulator.StepTo(simulator.get_context().get_time() + 0.01)


if __name__ == "__main__":
    main()  # This is what you would have, but the following is useful.

    # # These are temporary, for debugging, so meh for programming style.
    # import sys, trace
    #
    # # If there are segfaults, it's a good idea to always use stderr as it
    # # always prints to the screen, so you should get as much output as
    # # possible.
    # sys.stdout = sys.stderr
    #
    # # Now trace execution:
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.run('main()')
