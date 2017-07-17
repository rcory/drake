# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load("//tools:cpplint.bzl", "cpplint")
load(
    "//tools:drake.bzl",
    "drake_cc_library",
    "drake_cc_binary",
)

package(
    default_visibility = ["//visibility:public"],
)

drake_cc_library(
    name = "remy_controller",
    srcs = [
        "remy_controller.cc",
    ],
    hdrs = [
        "remy_controller.h",
    ],
    deps = [
        "//drake/examples/TRI_Remy:remy_common",
        "//drake/multibody:rigid_body_tree",
        "//drake/systems/framework",
    ],
)

cpplint()
