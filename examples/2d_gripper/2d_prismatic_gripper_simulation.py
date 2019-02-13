"""Provides an example of a 2d gripper, with prismatic fingertips."""

import argparse

from pydrake.common import FindResourceOrThrow
from pydrake.geometry import (ConnectDrakeVisualizer, SceneGraph)
from pydrake.lcm import DrakeLcm
from pydrake.multibody.tree import UniformGravityFieldElement
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix

from pydrake.systems.controllers import InverseDynamicsController

from pydrake.systems.primitives import ConstantVectorSource
from pydrake.systems.primitives import Demultiplexer
from pydrake.systems.primitives import Multiplexer
from pydrake.multibody.plant import ConnectContactResultsToDrakeVisualizer
from pydrake.systems.primitives import Sine
from pydrake.systems.primitives import Adder

import numpy as np


def weld_gripper_frames(gripper):
    outer_radius = 0.185
    f1_angle = 60*(np.pi/180.)

    XT = RigidTransform(RollPitchYaw([0, 0, 0]), [0, 0, outer_radius])

    # Weld the first finger
    # X_PC1 = RigidTransform(RollPitchYaw([0, 0, 0]), [0, 0, outer_radius])
    X_PC1 = RigidTransform(RollPitchYaw([f1_angle, 0, 0]), [0, 0, 0]).multiply(XT)
    child_frame = gripper.GetFrameByName("finger1_base")
    gripper.WeldFrames(gripper.world_frame(),
                       child_frame, X_PC1.GetAsIsometry3())

    # Weld the second finger
    X_PC2 = RigidTransform(
        RollPitchYaw([120*(np.pi/180.), 0, 0]), [0, 0, 0]).multiply(X_PC1)
    child_frame = gripper.GetFrameByName("finger2_base")
    gripper.WeldFrames(gripper.world_frame(),
                       child_frame, X_PC2.GetAsIsometry3())

    # Weld the 3rd finger
    X_PC3 = RigidTransform(
        RollPitchYaw([120*(np.pi/180.), 0, 0]), [0, 0, 0]).multiply(X_PC2)
    child_frame = gripper.GetFrameByName("finger3_base")
    gripper.WeldFrames(gripper.world_frame(),
                       child_frame, X_PC3.GetAsIsometry3())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target_realtime_rate", type=float, default=1.0,
        help="Desired rate relative to real time.  See documentation for "
             "Simulator::set_target_realtime_rate() for details.")
    parser.add_argument(
        "--simulation_time", type=float, default=3.0,
        help="Desired duration of the simulation in seconds.")
    parser.add_argument(
        "--time_step", type=float, default=1e-3,
        help="If greater than zero, the plant is modeled as a system with "
             "discrete updates and period equal to this time_step. "
             "If 0, the plant is modeled as a continuous system.")
    # Note: continuous system doesn't seem to model joint limits
    parser.add_argument(
        "--add_gravity", type=bool, default=True,
        help="Determines whether gravity is added to the simulation.")

    args = parser.parse_args()

    file_name = FindResourceOrThrow(
        "drake/examples/2d_gripper/planar_gripper.sdf")
    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    gripper = builder.AddSystem(MultibodyPlant(time_step=args.time_step))
    gripper.RegisterAsSourceForSceneGraph(scene_graph)
    gripper_id = Parser(plant=gripper).AddModelFromFile(file_name, "gripper")
    weld_gripper_frames(gripper)

    # Adds the object to be manipulated.
    # TODO(rcory) adding this object changes the estimated state dimension
    #   (affects the controller).
    object_file_name = FindResourceOrThrow(
        "drake/examples/2d_gripper/061_foam_brick.sdf")
    Parser(plant=gripper).AddModelFromFile(object_file_name, "object")

    # Create the controlled plant. Contains only the fingers (no objects).
    control_gripper = MultibodyPlant(time_step=args.time_step)
    control_gripper_id = \
        Parser(plant=control_gripper).AddModelFromFile(file_name, "gripper")
    weld_gripper_frames(control_gripper)

    # Add gravity?
    if args.add_gravity:
        gripper.AddForceElement(UniformGravityFieldElement())
        control_gripper.AddForceElement(UniformGravityFieldElement())

    # Finalize the plants.
    gripper.Finalize()
    control_gripper.Finalize()
    assert gripper.geometry_source_is_registered()

    # gripper.set_penetration_allowance(0.5)

# ===== Inverse Dynamics Source ============================================
    # Add an ID controller to hold the fingers in place.
    Kp = np.array([1, 1, 1, 1, 1, 1]) * 1500
    Kd = np.array([1, 1, 1, 1, 1, 1]) * 500
    Ki = np.array([1, 1, 1, 1, 1, 1]) * 500
    id_controller = builder.AddSystem(
        InverseDynamicsController(control_gripper, Kp, Ki, Kd, False))

    # Connect the ID controller
    # TODO(rcory) modify this when I add the object, since the state output will
    #  contain the object state, which the ID controller doesn't need)
    builder.Connect(gripper.get_continuous_state_output_port(gripper_id),
                    id_controller.get_input_port_estimated_state())

    # # Constant reference
    # x_ref = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # [q_ref, v_ref]
    #
    # # Connect the desired state
    # const_src = builder.AddSystem(ConstantVectorSource(x_ref))
    # builder.Connect(const_src.get_output_port(0),
    #                 id_controller.get_input_port_desired_state())

    # Sine reference
    amplitudes = [0.05, 0.05, 0.05, 0, 0, 0]
    # amplitudes = [0, 0, 0, 0, 0, 0]
    frequencies = [6, 6, 6, 6, 6, 6]
    phases = [0, 0, 0.5, 0, 0, 0]
    sine_source = builder.AddSystem(Sine(amplitudes, frequencies, phases))
    smux = builder.AddSystem(Multiplexer([6, 6]))  # [q, qdot]

    # add the offsets
    adder = builder.AddSystem(Adder(2, 6))
    offsets = builder.AddSystem(ConstantVectorSource([0, 0, 0, 0, -0.02, 0]))
    builder.Connect(sine_source.get_output_port(0),
                    adder.get_input_port(0))
    builder.Connect(offsets.get_output_port(0),
                    adder.get_input_port(1))

    # builder.Connect(sine_source.get_output_port(0), smux.get_input_port(0))
    builder.Connect(adder.get_output_port(0), smux.get_input_port(0))
    builder.Connect(sine_source.get_output_port(1), smux.get_input_port(1))
    builder.Connect(smux.get_output_port(0),
                    id_controller.get_input_port_desired_state())

    # TODO(rcory) This connect code doesn't work...seems indices don't match.
    # builder.Connect(id_controller.get_output_port_control(),
    #                 gripper.get_actuation_input_port())

    # This is the demux/mux hack
    demux = builder.AddSystem(Demultiplexer(6, 1))
    mux = builder.AddSystem(Multiplexer(6))
    builder.Connect(id_controller.get_output_port_control(),
                    demux.get_input_port(0))
    builder.Connect(demux.get_output_port(0), mux.get_input_port(0))
    builder.Connect(demux.get_output_port(1), mux.get_input_port(2))
    builder.Connect(demux.get_output_port(2), mux.get_input_port(4))
    builder.Connect(demux.get_output_port(3), mux.get_input_port(1))
    builder.Connect(demux.get_output_port(4), mux.get_input_port(3))
    builder.Connect(demux.get_output_port(5), mux.get_input_port(5))
    builder.Connect(mux.get_output_port(0),
                    gripper.get_actuation_input_port(gripper_id))

    # ==== For testing indices ============
    # cs = builder.AddSystem(ConstantVectorSource([0, 0, 0, 0.1, 0, 0]))
    # builder.Connect(cs.get_output_port(0),
    #                 gripper.get_actuation_input_port(gripper_id))

    # cs = builder.AddSystem(ConstantVectorSource([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0]))
    # builder.Connect(cs.get_output_port(0),
    #                 gripper.get_applied_generalized_force_input_port())
    # ======================================

    # TODO(rcory) This version doesn't work when the box is added
    # builder.Connect(id_controller.get_output_port_control(),
    #                 gripper.get_applied_generalized_force_input_port())
    #
    # # Plug the actuator input port
    # const_actuator_src = builder.AddSystem(ConstantVectorSource([0, 0, 0, 0, 0, 0]))
    # builder.Connect(const_actuator_src.get_output_port(0),
    #                 gripper.get_actuation_input_port())
    # ==========================================================================

    # Connect the SceneGraph with MBP.
    builder.Connect(
        scene_graph.get_query_output_port(),
        gripper.get_geometry_query_input_port())
    builder.Connect(
        gripper.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(gripper.get_source_id()))

    ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)

    # Publish contact results
    lcm = DrakeLcm()
    ConnectContactResultsToDrakeVisualizer(builder, gripper, lcm)

    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    gripper_context = diagram.GetMutableSubsystemContext(
        gripper, diagram_context)

    # gripper_context.FixInputPort(gripper.get_actuation_input_port().get_index(),
    #                              [0, 0, 0, 0, 0, 0])

    # Set the initial conditions.
    f1_slider = gripper.GetJointByName("finger2_ElbowJoint")
    f1_pin = gripper.GetJointByName("finger2_ShoulderJoint")
    f1_slider.set_translation(context=gripper_context, translation=0)
    f1_pin.set_angle(context=gripper_context, angle=0.)

    simulator = Simulator(diagram, diagram_context)
    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(args.target_realtime_rate)
    simulator.Initialize()

    # simulator.StepTo(args.simulation_time)  # <-- can't interrupt this version
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
