# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load(
    "//tools:drake.bzl",
    "drake_cc_binary",
)
load("//tools:lint.bzl", "add_lint_tests")

filegroup(
    name = "models",
    srcs = glob([
        "**/*.obj",
        "**/*.mtl",
        "**/*.urdf",
        "**/*.txt",
    ]),
    visibility = ["//visibility:public"],
)

drake_cc_binary(
    name = "box_rotation_demo",
    srcs = [
        "box_rotation_demo.cc",
    ],
    data = [
        "//drake/examples/kuka_iiwa_arm/dev/box_rotation:models",
    ],
    deps = [
        "//drake/common:find_resource",
        "//drake/examples/kuka_iiwa_arm:iiwa_common",
        "//drake/examples/kuka_iiwa_arm:iiwa_lcm",
        "//drake/examples/kuka_iiwa_arm/pick_and_place",
        "//drake/util:lcm_util",
    ],
)

# === test/ ===

add_lint_tests()
