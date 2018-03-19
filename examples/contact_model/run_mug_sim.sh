#!/usr/bin/env bash
# -- Original Parameters --
#${DRAKE_WORKSPACE}/bazel-bin/examples/contact_model/rigid_mug_gripper \
#--simulation_type=compliant --ts_dt=1e-3 --rk_dt=1e-4 --accuracy=5e-5 \
#--ring_samples=4 --pad_depth=4e-3 --ring_orient=0 --ring_youngs_modulus=-1 \
#--ring_dissipation=-1 --ring_static_friction=-1 --ring_dynamic_friction=-1 \
#--px=0 --py=0 --pz=0 --rx=0 --ry=0 --rz=0


${DRAKE_WORKSPACE}/bazel-bin/examples/contact_model/rigid_mug_gripper \
--simulation_type=compliant --ts_dt=1e-3 \
--rk_type=rk2 rk_dt=1e-4 --accuracy=5e-5 \
--ring_samples=4 --pad_depth=4e-3 --ring_orient=0 --ring_youngs_modulus=-1 \
--ring_dissipation=-1 --ring_static_friction=-1 --ring_dynamic_friction=-1 \
--px=0 --py=0 --pz=0 --rx=0 --ry=0 --rz=0
