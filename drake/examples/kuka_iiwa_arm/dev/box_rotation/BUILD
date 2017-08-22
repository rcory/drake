# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load(
    "//tools:drake.bzl",
    "drake_cc_binary",
)
load("//tools:lint.bzl", "add_lint_tests")

filegroup(
    name = "mydata",
    srcs = glob([
        "**/*.urdf",
        "**/*.txt",
    ]),
    visibility = ["//visibility:public"],
)

drake_cc_binary(
    name = "box_rotation_demo",
    srcs = ["box_rotation_demo.cc"],
    data = [
        ":mydata",
        "//drake/manipulation/models/iiwa_description:models",
    ],
    deps = [
        "//drake/common:find_resource",
        "//drake/examples/kuka_iiwa_arm:iiwa_common",
        "//drake/examples/kuka_iiwa_arm:iiwa_lcm",
        "//drake/examples/kuka_iiwa_arm/pick_and_place",
        "//drake/util:lcm_util",
    ],
)

drake_cc_binary(
    name = "iiwa_box_simulation",
    srcs = ["iiwa_box_simulation.cc"],
    add_test_rule = 1,
    data = [
        ":mydata",
        "//drake/examples/kuka_iiwa_arm:models",
        "//drake/manipulation/models/iiwa_description:models",
        "//drake/manipulation/models/wsg_50_description:models",
    ],
    test_rule_args = ["--simulation_sec=0.01"],
    deps = [
        "//drake/examples/kuka_iiwa_arm:iiwa_common",
        "//drake/examples/kuka_iiwa_arm:iiwa_lcm",
        "//drake/examples/kuka_iiwa_arm:oracular_state_estimator",
        "//drake/examples/kuka_iiwa_arm/iiwa_world:iiwa_wsg_diagram_factory",
        "//drake/examples/kuka_iiwa_arm/iiwa_world:world_sim_tree_builder",
        "//drake/lcm",
        "//drake/manipulation/schunk_wsg:schunk_wsg_constants",
        "//drake/manipulation/schunk_wsg:schunk_wsg_controller",
        "//drake/manipulation/schunk_wsg:schunk_wsg_lcm",
        "//drake/multibody/rigid_body_plant",
        "//drake/systems/analysis",
        "//drake/systems/controllers:inverse_dynamics_controller",
        "//drake/systems/controllers:pid_controller",
        "//drake/systems/primitives:constant_vector_source",
        "//drake/systems/primitives:matrix_gain",
        "//drake/util:lcm_util",
        "@com_github_gflags_gflags//:gflags",
    ],
)

# === test/ ===

add_lint_tests()
