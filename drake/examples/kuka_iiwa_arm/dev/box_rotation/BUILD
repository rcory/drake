# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load("//tools:cpplint.bzl", "cpplint")
load(
    "//tools:drake.bzl",
    "drake_cc_library",
    "drake_cc_binary",
)

filegroup(
    name = "mymodels",
    srcs = glob([
        "**/*.obj",
        "**/*.sdf",
        "**/*.urdf",
        "**/*.xml",
        "simple_keyframes.txt",
    ]),
    visibility = ["//visibility:public"],
)

drake_cc_binary(
    name = "dual_arm_hug_box",
    srcs = ["dual_arm_hug_box.cc"],
    data = [
        "//drake/examples/kuka_iiwa_arm:models",
        "//drake/examples/kuka_iiwa_arm/dev/box_rotation:mymodels",
        "//drake/manipulation/models/iiwa_description:models",
    ],
    deps = [
        "//drake/examples/kuka_iiwa_arm:iiwa_common",
        "//drake/examples/kuka_iiwa_arm:sim_diagram_builder",
        "//drake/examples/kuka_iiwa_arm/iiwa_world:world_sim_tree_builder",
        "//drake/lcm",
        "//drake/lcmtypes:contact_info_for_viz",
        "//drake/lcmtypes:contact_results_for_viz",
        "//drake/multibody/rigid_body_plant:contact_results_to_lcm",
        "//drake/systems/analysis",
        "//drake/systems/controllers:inverse_dynamics_controller",
        "//drake/systems/primitives:trajectory_source",
        "@lcmtypes_bot2_core",
        "@lcmtypes_robotlocomotion",
    ],
)

cpplint()
