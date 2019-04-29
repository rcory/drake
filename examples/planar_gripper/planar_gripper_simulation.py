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


class ActuatorTranslator(LeafSystem):
    def __init__(self, ordering):
        LeafSystem.__init__(self)
        self.ordering = ordering
        size = len(ordering)
        self._DeclareVectorInputPort("input1", BasicVector(size))
        self._DeclareVectorOutputPort("output1", BasicVector(size),
                                      self._reorder_output)

    def _reorder_output(self, context, output_vector):
        output_value = output_vector.get_mutable_value()
        output_value[:] = 0
        input_value = self.EvalVectorInput(context, 0).get_value()

        size = len(self.ordering)
        for i in range(size):
            output_value[i] = input_value[self.ordering[i]]


# def print_names(plant):
# # Print actuator names
#     for i in range(0, plant.num_actuators()):
#         print str(i) + ": " + plant.get_joint_actuator(JointActuatorIndex(i)).name()
#
#     print ""
#
#     # Print joint names
#     for i in range(0, plant.num_joints()):
#         print str(i) + ": " + plant.get_joint(JointIndex(i)).name()
#
#     print plant.num_positions()


# def get_control_port_mapping(plant, control_plant):
#
#     joint_ordering_names = [str]*control_plant.num_joints()
#
#     for i in range(0, control_plant.num_joints()):
#         joint_ordering_names[i] = \
#             str(control_plant.get_joint(JointIndex(i)).name())
#
#     joint_index_mapping = [None]*plant.num_joints()
#     for i in range(0, len(joint_ordering_names)):
#         joint_index_mapping[i] = plant.GetJointByName(joint_ordering_names[i]).index()
#
#     Sx = plant.MakeStateSelectorMatrix(joint_index_mapping)
#     Sy = plant.MakeActuatorSelectorMatrix(joint_index_mapping)
#
#     return joint_index_mapping


def weld_gripper_frames(plant):
    outer_radius = 0.19  # 19 - 22
    f1_angle = 60*(np.pi/180.)

    XT = RigidTransform(RollPitchYaw([0, 0, 0]), [0, 0, outer_radius])

    # Weld the first finger
    # X_PC1 = RigidTransform(RollPitchYaw([0, 0, 0]), [0, 0, outer_radius])
    X_PC1 = RigidTransform(RollPitchYaw([f1_angle, 0, 0]), [0, 0, 0]).multiply(XT)
    child_frame = plant.GetFrameByName("finger1_base")
    plant.WeldFrames(plant.world_frame(),
                       child_frame, X_PC1.GetAsIsometry3())

    # Weld the second finger
    X_PC2 = RigidTransform(
        RollPitchYaw([120*(np.pi/180.), 0, 0]), [0, 0, 0]).multiply(X_PC1)
    child_frame = plant.GetFrameByName("finger2_base")
    plant.WeldFrames(plant.world_frame(),
                       child_frame, X_PC2.GetAsIsometry3())

    # Weld the 3rd finger
    X_PC3 = RigidTransform(
        RollPitchYaw([120*(np.pi/180.), 0, 0]), [0, 0, 0]).multiply(X_PC2)
    child_frame = plant.GetFrameByName("finger3_base")
    plant.WeldFrames(plant.world_frame(),
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
        "drake/examples/planar_gripper/planar_gripper.sdf")
    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    plant = builder.AddSystem(MultibodyPlant(time_step=args.time_step))
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    plant_id = Parser(plant=plant).AddModelFromFile(file_name, "gripper")
    weld_gripper_frames(plant)

    # Adds the object to be manipulated.
    # TODO(rcory) adding this object changes the estimated state dimension
    #   (affects the controller).
    object_file_name = FindResourceOrThrow(
        "drake/examples/planar_gripper/brick.sdf")
    object_id = Parser(plant=plant).AddModelFromFile(object_file_name, "object")

    # Create the controlled plant. Contains only the fingers (no objects).
    control_plant = MultibodyPlant(time_step=args.time_step)
    control_plant_id = \
        Parser(plant=control_plant).AddModelFromFile(file_name, "gripper")
    weld_gripper_frames(control_plant)

    # Add gravity?
    if args.add_gravity:
        plant.AddForceElement(UniformGravityFieldElement())
        control_plant.AddForceElement(UniformGravityFieldElement())

    # Finalize the plants.
    plant.Finalize()
    control_plant.Finalize()
    assert plant.geometry_source_is_registered()

# ===== Inverse Dynamics Source ============================================
    # Add an ID controller to hold the fingers in place.
    Kp = np.array([1, 1, 1, 1, 1, 1]) * 1500
    Kd = np.array([1, 1, 1, 1, 1, 1]) * 500
    Ki = np.array([1, 1, 1, 1, 1, 1]) * 500
    id_controller = builder.AddSystem(
        InverseDynamicsController(control_plant, Kp, Ki, Kd, False))

    # Connect the ID controller
    # TODO(rcory) modify this when I add the object, since the state output will
    #  contain the object state, which the ID controller doesn't need)
    builder.Connect(plant.get_continuous_state_output_port(plant_id),
                    id_controller.get_input_port_estimated_state())

    # # Constant reference
    # x_ref = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # [q_ref, v_ref]
    #
    # # Connect the desired state
    # const_src = builder.AddSystem(ConstantVectorSource(x_ref))
    # builder.Connect(const_src.get_output_port(0),
    #                 id_controller.get_input_port_desired_state())

    # Sine reference
    amplitudes = [0.03, 0.03, 0.03, 0.05, 0.05, 0.05]
    # amplitudes = [0, 0, 0, 0, 0, 0]
    frequencies = [6, 6.1, 6.2, 6.3, 6.4, 6.5]
    phases = [0, 0.2, 0.5, 0, 0, 0]
    sine_source = builder.AddSystem(Sine(amplitudes, frequencies, phases))
    smux = builder.AddSystem(Multiplexer([6, 6]))  # [q, qdot]

    # Add Sine offsets
    # Order is [l1_sh, l2_sh, l3_sh, l1_el, l2_el, l3_el]
    # where sh: shoulder, el: elbow
    adder = builder.AddSystem(Adder(2, 6))
    offsets = builder.AddSystem(
        ConstantVectorSource([-0.65, -0.5, 0.65, 1.0, 0.95, -1.0]))
    builder.Connect(sine_source.get_output_port(0),
                    adder.get_input_port(0))
    builder.Connect(offsets.get_output_port(0),
                    adder.get_input_port(1))

    # Connect the offset Sine reference to the IDC reference input
    builder.Connect(adder.get_output_port(0), smux.get_input_port(0))
    builder.Connect(sine_source.get_output_port(1), smux.get_input_port(1))
    builder.Connect(smux.get_output_port(0),
                    id_controller.get_input_port_desired_state())

    # TODO(rcory) This connect code doesn't work...seems indices don't match.
    # builder.Connect(id_controller.get_output_port_control(),
    #                 plant.get_actuation_input_port())

    # Hack needed to map the ID controller outputs to MBP inputs.
    # TODO(rcory) Update ID controller to be "smarter" and know about the
    #  required index actuator ordering going into MBP.
    translator = builder.AddSystem(ActuatorTranslator([0, 3, 1, 4, 2, 5]))
    builder.Connect(id_controller.get_output_port_control(),
                    translator.get_input_port(0))
    builder.Connect(translator.get_output_port(0),
                    plant.get_actuation_input_port(plant_id))

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

    # plant_context.FixInputPort(plant.get_actuation_input_port().get_index(),
    #                              [0, 0, 0, 0, 0, 0])

    # Set the plant initial conditions.
    sh_pin = plant.GetJointByName("finger1_ShoulderJoint")
    el_pin = plant.GetJointByName("finger1_ElbowJoint")
    sh_pin.set_angle(context=plant_context, angle=-0.65)
    el_pin.set_angle(context=plant_context, angle=1.0)

    # Set the plant initial conditions.
    sh_pin = plant.GetJointByName("finger2_ShoulderJoint")
    el_pin = plant.GetJointByName("finger2_ElbowJoint")
    sh_pin.set_angle(context=plant_context, angle=-0.5)
    el_pin.set_angle(context=plant_context, angle=0.95)

    # Set the plant initial conditions.
    sh_pin = plant.GetJointByName("finger3_ShoulderJoint")
    el_pin = plant.GetJointByName("finger3_ElbowJoint")
    sh_pin.set_angle(context=plant_context, angle=0.65)
    el_pin.set_angle(context=plant_context, angle=-1.0)

    # Set the box initial conditions
    X_WObj = RigidTransform(RollPitchYaw([0, 0, 0]), [0, 0, 0])
    body_index_vec = plant.GetBodyIndices(object_id)
    box_body = plant.get_body(body_index_vec[0])
    plant.SetFreeBodyPose(plant_context, box_body, X_WObj.GetAsIsometry3())

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
