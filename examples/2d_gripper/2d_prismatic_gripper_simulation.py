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

#from pydrake.systems.primitives import Sine

import numpy as np

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

    if args.add_gravity:
        gripper.AddForceElement(UniformGravityFieldElement())

    # add the object
    object_file_name = FindResourceOrThrow(
        "drake/examples/2d_gripper/061_foam_brick.sdf")
    object_model = \
        Parser(plant=gripper).AddModelFromFile(object_file_name, "object")

    # Weld the 1st finger
    finger1_model = Parser(plant=gripper).AddModelFromFile(file_name, "finger1")
    X_PC1 = RigidTransform(RollPitchYaw([0, 0, 0]), [0, 0, 0.18])
    child_frame = gripper.GetFrameByName("ground", finger1_model)
    gripper.WeldFrames(gripper.world_frame(),
                       child_frame, X_PC1.GetAsIsometry3())

    # Weld the 2nd finger
    finger2_model = Parser(plant=gripper).AddModelFromFile(file_name, "finger2")
    X_PC2 = RigidTransform(
        RollPitchYaw([120*(np.pi/180.), 0, 0]), [0, 0, 0]).multiply(X_PC1)
    child_frame = gripper.GetFrameByName("ground", finger2_model)
    gripper.WeldFrames(gripper.world_frame(),
                       child_frame, X_PC2.GetAsIsometry3())

    # Weld the 3rd finger
    finger3_model = Parser(plant=gripper).AddModelFromFile(file_name, "finger3")
    X_PC3 = RigidTransform(
        RollPitchYaw([120*(np.pi/180.), 0, 0]), [0, 0, 0]).multiply(X_PC2)
    child_frame = gripper.GetFrameByName("ground", finger3_model)
    gripper.WeldFrames(gripper.world_frame(),
                       child_frame, X_PC3.GetAsIsometry3())

    gripper.Finalize()
    assert gripper.geometry_source_is_registered()

    # Connect the SceneGraph with MBP.
    builder.Connect(
        scene_graph.get_query_output_port(),
        gripper.get_geometry_query_input_port())
    builder.Connect(
        gripper.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(gripper.get_source_id()))

    # Add a sine wave source to the actuator inputs.
    # This code is WIP.
    # sine_source = builder.AddSystem(Sine(0.5, 1.0, 0.0, 3))

    ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)

    # Publish contact results
    lcm = DrakeLcm()
    contact_results_to_lcm = builder.AddSystem(
        ContactResultsToLcmSystem(gripper))

    # # Long version...
    # lcm_pub_system = LcmPublisherSystem.Make(
    #     channel="CONTACT_RESULTS", lcm_type=lcmt_contact_results_for_viz,
    #     lcm=lcm, publish_period=1.0/30.0, use_cpp_serializer=True)
    # contact_results_publisher = builder.AddSystem(lcm_pub_system)

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

    gripper_context.FixInputPort(
        gripper.get_actuation_input_port(finger1_model).get_index(), [0, 0])
    gripper_context.FixInputPort(
        gripper.get_actuation_input_port(finger2_model).get_index(), [0, 0])
    gripper_context.FixInputPort(
        gripper.get_actuation_input_port(finger3_model).get_index(), [0, 0])


    # link2_slider = gripper.GetJointByName("ElbowJoint", finger1_model)
    # link1_pin = gripper.GetJointByName("ShoulderJoint", finger1_model)
    # link2_slider.set_translation(context=gripper_context, translation=0.)
    # link1_pin.set_angle(context=gripper_context, angle=0.)

    simulator = Simulator(diagram, diagram_context)
    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(args.target_realtime_rate)
    simulator.Initialize()
    simulator.StepTo(args.simulation_time)


if __name__ == "__main__":
    main()
