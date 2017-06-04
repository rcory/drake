# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load("//tools:cpplint.bzl", "cpplint")
load(
    "//tools:drake.bzl",
    "drake_cc_binary",
    "drake_cc_library",
)

drake_cc_library(
    name = "dual_arms_rotate_box_planner",
    srcs = ["dual_arms_rotate_box_planner.cc"],
    hdrs = ["dual_arms_rotate_box_planner.h"],
    deps = [
        "//drake/lcm",
        "//drake/multibody:rigid_body_tree",
        "//drake/multibody/parsers",
        "//drake/multibody/rigid_body_plant:drake_visualizer",
        "//drake/solvers:mathematical_program",
    ],
)

drake_cc_binary(
    name = "dual_arms_rotate_box_demo",
    srcs = ["dual_arms_rotate_box_demo.cc"],
    deps = [
        "dual_arms_rotate_box_planner",
    ],
)
