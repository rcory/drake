# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load("//tools:cpplint.bzl", "cpplint")
load(
    "//tools:drake.bzl",
    "drake_cc_library",
    "drake_cc_binary",
    "drake_cc_googletest",
)

filegroup(
    name = "models",
    srcs = glob([
        "**/*.obj",
        "**/*.mtl",
        "**/*.urdf",
    ]),
    visibility = ["//visibility:public"],
)

package(
    default_visibility = [":__subpackages__"],
)

drake_cc_library(
    name = "remy_common",
    srcs = [
        "remy_common.cc",
    ],
    hdrs = [
        "remy_common.h",
    ],
    deps = [
        "//drake/multibody:rigid_body_tree",
        "//drake/multibody/parsers",
    ],
)

drake_cc_binary(
    name = "run_passive_remy_demo",
    srcs = ["run_passive_remy_demo.cc"],
    data = [
        "//drake/examples/TRI_Remy:models",
        "//drake/manipulation/models/jaco_description:models",
    ],
    deps = [
        ":remy_common",
        "//drake/common:text_logging_gflags",
        "//drake/lcm",
        "//drake/multibody:rigid_body_tree_construction",
        "//drake/multibody/rigid_body_plant",
        "//drake/multibody/rigid_body_plant:drake_visualizer",
        "//drake/systems/analysis:simulator",
        "//drake/systems/primitives:constant_vector_source",
        "@com_github_gflags_gflags//:gflags",
    ],
)

#drake_cc_googletest(
#    name = "remy_simulation_test",
#    srcs = ["test/remy_simulation_test.cc"],
#    data = [
#        "//drake/examples/TRI_Remy:models",
#    ],
#    deps = [
#        ":remy_common",
#        "//drake/common:text_logging_gflags",
#        "//drake/lcm",
#        "//drake/multibody:rigid_body_tree_construction",
#        "//drake/multibody/rigid_body_plant",
#        "//drake/multibody/rigid_body_plant:drake_visualizer",
#        "//drake/systems/analysis:simulator",
#        "//drake/systems/primitives:constant_vector_source",
#    ],
#)

cpplint()
