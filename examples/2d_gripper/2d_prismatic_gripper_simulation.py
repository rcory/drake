"""Provides an example of a 2d gripper, with prismatic fingertips."""

import argparse

from pydrake.common import FindResourceOrThrow
from pydrake.geometry import (ConnectDrakeVisualizer, SceneGraph)
from pydrake.lcm import DrakeLcm
from pydrake.systems.lcm import LcmPublisherSystem
from pydrake.multibody.multibody_tree import UniformGravityFieldElement
from pydrake.multibody.multibody_tree.multibody_plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.multibody_tree.multibody_plant import ContactResultsToLcmSystem

from drake import lcmt_contact_results_for_viz
from pydrake.systems.controllers import InverseDynamicsController

from pydrake.systems.primitives import Sine
from pydrake.systems.primitives import ConstantVectorSource
from pydrake.systems.primitives import Demultiplexer
from pydrake.systems.primitives import Multiplexer

import numpy as np


def add_gripper_model(gripper, file_name):
    # Weld the first finger
    finger1 = Parser(plant=gripper).AddModelFromFile(file_name, "finger1")
    X_PC1 = RigidTransform(RollPitchYaw([0, 0, 0]), [0, 0, 0.18])
    child_frame = gripper.GetFrameByName("ground", finger1)
    gripper.WeldFrames(gripper.world_frame(),
                       child_frame, X_PC1.GetAsIsometry3())

    # Weld the 2nd finger
    finger2 = Parser(plant=gripper).AddModelFromFile(file_name, "finger2")
    X_PC2 = RigidTransform(
        RollPitchYaw([120*(np.pi/180.), 0, 0]), [0, 0, 0]).multiply(X_PC1)
    child_frame = gripper.GetFrameByName("ground", finger2)
    gripper.WeldFrames(gripper.world_frame(),
                       child_frame, X_PC2.GetAsIsometry3())

    # Weld the 3rd finger
    finger3 = Parser(plant=gripper).AddModelFromFile(file_name, "finger3")
    X_PC3 = RigidTransform(
        RollPitchYaw([120*(np.pi/180.), 0, 0]), [0, 0, 0]).multiply(X_PC2)
    child_frame = gripper.GetFrameByName("ground", finger3)
    gripper.WeldFrames(gripper.world_frame(),
                       child_frame, X_PC3.GetAsIsometry3())

    return finger1, finger2, finger3


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
        "--add_gravity", type=bool, default=False,
        help="Determines whether gravity is added to the simulation.")

    args = parser.parse_args()

    file_name = FindResourceOrThrow(
        "drake/examples/2d_gripper/2d_prismatic_finger.sdf")
    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    gripper = builder.AddSystem(MultibodyPlant(time_step=args.time_step))
    gripper.RegisterAsSourceForSceneGraph(scene_graph)

    # Adds the object to be manipulated.
    # TODO(rcory) adding this object changes the state dimension
    #   (affects the controller).
    # object_file_name = FindResourceOrThrow(
    #     "drake/examples/2d_gripper/061_foam_brick.sdf")
    # object_model = \
    #     Parser(plant=gripper).AddModelFromFile(object_file_name, "object")

    # Add the fingers to the simulation plant.
    (finger1_id, finger2_id, finger3_id) = \
        add_gripper_model(gripper, file_name)

    # Create the controlled plant. Contains only the fingers (no objects).
    control_gripper = MultibodyPlant(time_step=args.time_step)

    # Add the fingers to the control plant.
    (control_finger1_id, control_finger2_id, control_finger3_id) = \
        add_gripper_model(control_gripper, file_name)

    # Add gravity?
    if args.add_gravity:
        gripper.AddForceElement(UniformGravityFieldElement())
        control_gripper.AddForceElement(UniformGravityFieldElement())

    # Finalize the plants.
    gripper.Finalize()
    control_gripper.Finalize()
    assert gripper.geometry_source_is_registered()

    # Connect the SceneGraph with MBP.
    builder.Connect(
        scene_graph.get_query_output_port(),
        gripper.get_geometry_query_input_port())
    builder.Connect(
        gripper.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(gripper.get_source_id()))

    # ===== Sine Source ========================================================
    # Add a sine wave source to the actuator inputs.
    # This code is WIP.
    # sine_source1 = builder.AddSystem(Sine(.005, 0.5, 0.0, 2))
    # sine_source2 = builder.AddSystem(Sine(0.5, 1.0, 0.0, 2))
    # sine_source3 = builder.AddSystem(Sine(0.5, 1.0, 0.0, 2))

    # builder.Connect(sine_source1.get_output_port(0),
    #                 gripper.get_actuation_input_port(finger1_id))
    # ==========================================================================

    # ===== Constant Source ====================================================
    # # Adds a constant source to the first finger.
    # const_src = builder.AddSystem(ConstantVectorSource([0.05, 0]))
    # builder.Connect(const_src.get_output_port(0),
    #                 gripper.get_actuation_input_port(finger1_id))
    # ==========================================================================

    # ===== Inverse Dynamics Source ============================================
    # Add an ID controller to hold the fingers in place.
    Kp = np.array([1, 1, 1, 1, 1, 1]) * 50
    Kd = np.array([1, 1, 1, 1, 1, 1]) * 10
    Ki = np.array([1, 1, 1, 1, 1, 1]) * 10
    id_controller = builder.AddSystem(
        InverseDynamicsController(control_gripper, Kp, Ki, Kd, False))

    # Connect the ID controller
    # TODO(rcory) modify this when I add the object, since the state output will
    #  contain the object state, which the ID controller doesn't need)
    builder.Connect(gripper.get_continuous_state_output_port(),
                    id_controller.get_input_port_estimated_state())

    # TODO(rcory) This desired state ordering is odd. E.g., the q's are ordered
    #  as follows:
    #  [model1_q1, model2_q1, ..., model1_q2, model2_q2,...]
    #  The same is true for the velocities.
    desired_state = [0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Connect the desired state
    const_src = builder.AddSystem(ConstantVectorSource(desired_state))
    builder.Connect(const_src.get_output_port(0),
                    id_controller.get_input_port_desired_state())

    # TODO(rcory) These mux gymnastics are a hack. I shouldn't need this.
    demux = builder.AddSystem(Demultiplexer(6, 1))
    mux1 = builder.AddSystem(Multiplexer(2))
    mux2 = builder.AddSystem(Multiplexer(2))
    mux3 = builder.AddSystem(Multiplexer(2))

    builder.Connect(id_controller.get_output_port_control(),
                    demux.get_input_port(0))

    builder.Connect(demux.get_output_port(0), mux1.get_input_port(0))
    builder.Connect(demux.get_output_port(3), mux1.get_input_port(1))

    builder.Connect(demux.get_output_port(1), mux2.get_input_port(0))
    builder.Connect(demux.get_output_port(4), mux2.get_input_port(1))

    builder.Connect(demux.get_output_port(2), mux3.get_input_port(0))
    builder.Connect(demux.get_output_port(5), mux3.get_input_port(1))

    builder.Connect(mux1.get_output_port(0),
                    gripper.get_actuation_input_port(finger1_id))
    builder.Connect(mux2.get_output_port(0),
                    gripper.get_actuation_input_port(finger2_id))
    builder.Connect(mux3.get_output_port(0),
                    gripper.get_actuation_input_port(finger3_id))

    # TODO(rcory) The code below should work, but the id output ordering is off.
    #  See ordering comment above.
    # Demux the torque commands into separate fingers (each is a different
    # model instance in MBP).
    # demux = builder.AddSystem(Demultiplexer(6, 2))
    # builder.Connect(id_controller.get_output_port_control(),
    #                 demux.get_input_port(0))
    #
    # builder.Connect(demux.get_output_port(0),
    #                 gripper.get_actuation_input_port(finger1_id))
    # builder.Connect(demux.get_output_port(1),
    #                 gripper.get_actuation_input_port(finger2_id))
    # builder.Connect(demux.get_output_port(2),
    #                 gripper.get_actuation_input_port(finger3_id))
    # ==========================================================================

    ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)

    # Publish contact results
    lcm = DrakeLcm()
    contact_results_to_lcm = builder.AddSystem(
        ContactResultsToLcmSystem(gripper))

    # Connect contact result to LCM.
    contact_results_publisher = builder.AddSystem(
        LcmPublisherSystem.Make(channel="CONTACT_RESULTS",
                                lcm_type=lcmt_contact_results_for_viz, lcm=lcm,
                                publish_period=1.0/30.0,
                                use_cpp_serializer=True))

    # Connect contact results to lcm msg.
    builder.Connect(gripper.get_contact_results_output_port(),
                    contact_results_to_lcm.get_contact_result_input_port())
    builder.Connect(contact_results_to_lcm.get_lcm_message_output_port(),
                    contact_results_publisher.get_input_port(0))

    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    gripper_context = diagram.GetMutableSubsystemContext(
        gripper, diagram_context)

    # Fix the actuation ports to zero.
    # gripper_context.FixInputPort(
    #     gripper.get_actuation_input_port(finger1_id).get_index(), [0, 0])
    # gripper_context.FixInputPort(
    #     gripper.get_actuation_input_port(finger2_id).get_index(), [0, 0])
    # gripper_context.FixInputPort(
    #     gripper.get_actuation_input_port(finger3_id).get_index(), [0, 0])

    # Set the initial conditions.
    # link2_slider = gripper.GetJointByName("ElbowJoint", finger1_model)
    # link1_pin = gripper.GetJointByName("ShoulderJoint", finger1_model)
    # link2_slider.set_translation(context=gripper_context, translation=0.)
    # link1_pin.set_angle(context=gripper_context, angle=0.)

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
