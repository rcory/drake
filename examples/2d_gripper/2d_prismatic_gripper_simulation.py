"""Provides an example of a 2d gripper, with prismatic fingertips."""

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

from pydrake.systems.controllers import InverseDynamicsController

from pydrake.systems.primitives import ConstantVectorSource

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

    return finger1, finger2


def main():
    file_name = FindResourceOrThrow(
        "drake/examples/2d_gripper/2d_prismatic_finger.sdf")
    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    gripper = builder.AddSystem(MultibodyPlant(0.001))
    gripper.RegisterAsSourceForSceneGraph(scene_graph)

    # Add the fingers to the simulation plant.
    (finger1_id, finger2_id) = add_gripper_model(gripper, file_name)

    gripper.AddForceElement(UniformGravityFieldElement())

    # Finalize the plants.
    gripper.Finalize()
    assert gripper.geometry_source_is_registered()

    # Connect the SceneGraph with MBP.
    builder.Connect(
        scene_graph.get_query_output_port(),
        gripper.get_geometry_query_input_port())
    builder.Connect(
        gripper.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(gripper.get_source_id()))

    # Add an ID controller to hold the fingers in place.
    Kp = np.array([1, 1, 1, 1]) * 50
    Kd = np.array([1, 1, 1, 1]) * 10
    Ki = np.array([1, 1, 1, 1]) * 10
    id_controller = builder.AddSystem(
        InverseDynamicsController(gripper, Kp, Ki, Kd, False))

    # Connect the ID controller
    builder.Connect(gripper.get_continuous_state_output_port(),
                    id_controller.get_input_port_estimated_state())

    desired_state = [0, 0, 0, 0, 0, 0, 0, 0]

    # Connect the desired state
    const_src = builder.AddSystem(ConstantVectorSource(desired_state))
    builder.Connect(const_src.get_output_port(0),
                    id_controller.get_input_port_desired_state())

    # Connect the single actuation port
    builder.Connect(id_controller.get_output_port_control(),
                    gripper.get_generalized_forces_input_port())

    ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)

    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    gripper_context = diagram.GetMutableSubsystemContext(
        gripper, diagram_context)

    # Fix the model instance actuation ports to zero. These are added to the
    # single vector actuation input port returned by get_actuation_input_port().
    # Produces errors if these ports are left unconnected.
    gripper_context.FixInputPort(
        gripper.get_actuation_input_port(finger1_id).get_index(), [0, 0])
    gripper_context.FixInputPort(
        gripper.get_actuation_input_port(finger2_id).get_index(), [0, 0])

    simulator = Simulator(diagram, diagram_context)
    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    while simulator.get_context().get_time() < 3.0:
        simulator.StepTo(simulator.get_context().get_time() + 0.01)


if __name__ == "__main__":
    main()  # This is what you would have, but the following is useful.