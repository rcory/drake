# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load("//tools:cpplint.bzl", "cpplint")
load(
    "//tools:drake.bzl",
    "drake_cc_binary",
    "drake_cc_library",
    "drake_cc_googletest",
)

#drake_cc_library(
#    name = "dual_arms_rotate_box_planner",
#    srcs = ["dual_arms_rotate_box_planner.cc"],
#    hdrs = [
#        "dual_arms_rotate_box_planner.h",
#    ],
#    deps = [
#        "//drake/common:drake_path",
#        "//drake/lcm",
#        "//drake/multibody:rigid_body_tree",
#        "//drake/multibody/parsers",
#        "//drake/multibody/rigid_body_plant:drake_visualizer",
#        "//drake/solvers:mathematical_program",
#    ],
#)

drake_cc_library(
    name = "dual_arms_box_rotation_planner",
    srcs = ["dual_arms_box_rotation_planner.cc"],
    hdrs = ["dual_arms_box_rotation_planner.h"],
    deps = [
        ":dual_arms_box_util",
        "//drake/multibody:inverse_kinematics",
        "//drake/multibody:rigid_body_constraint",
    ],
)

#drake_cc_googletest(
#    name = "dual_arms_rotate_box_planner_test",
#    deps = [
#        ":dual_arms_rotate_box_planner",
#    ],
#)

drake_cc_library(
    name = "dual_arms_box_util",
    srcs = [
        "dual_arms_box_util.cc",
    ],
    hdrs = [
        "dual_arms_box_util.h",
    ],
    data = ["//drake/examples/kuka_iiwa_arm/dev/box_rotation:mydata"],
    deps = [
        "//drake/common:find_resource",
        "//drake/lcm",
        "//drake/multibody:rigid_body_tree",
        "//drake/multibody/parsers",
        "//drake/multibody/rigid_body_plant:drake_visualizer",
    ],
)

#drake_cc_binary(
#    name = "dual_arms_box_ik_planner",
#    srcs = ["dual_arms_box_ik_planner.cc"],
#    deps = [
#        "dual_arms_box_util",
#        "//drake/manipulation/util:simple_tree_visualizer",
#        "//drake/multibody:inverse_kinematics",
#    ],
#)
#
#drake_cc_binary(
#    name = "dual_arms_box_rotation_planner_test",
#    srcs = ["test/dual_arms_box_rotation_planner_test.cc"],
#    deps = [
#        "dual_arms_box_rotation_planner",
#        "dual_arms_box_util",
#        "//drake/manipulation/util:simple_tree_visualizer",
#    ],
#)

drake_cc_binary(
    name = "dual_arms_box_planner_demo",
    srcs = ["test/dual_arms_box_planner_demo.cc"],
    data = [
        "//drake/manipulation/models/iiwa_description:models",
    ],
    deps = [
        "dual_arms_box_rotation_planner",
        "dual_arms_box_util",
        "//drake/examples/kuka_iiwa_arm:iiwa_common",
        "//drake/examples/kuka_iiwa_arm/pick_and_place",
        "//drake/manipulation/util:simple_tree_visualizer",
        "@optitrack_driver//lcmtypes:optitrack_lcmtypes",
    ],
)
