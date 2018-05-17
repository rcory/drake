#!/usr/bin/env bash

# system_type can be "continuous" or "discrete"
# rk_type: can be "rk2" or "rk3"
# ts_dt: is the dt used for the discrete (a.k.a. time-stepping) system sim
# rk_dt: is the dt used for the continuous system sim.
#
# Working parameters
# ** rk2: rk_dt=6e-6 (based on 8 sphere pad), gripper_force=150
# ** rk3: rk_dt=1e-3, accuracy=1e-3, gripper_force=150
# ** discrete: ts_dt=1e-3, gripper_force=150

${DRAKE_WORKSPACE}/bazel-bin/examples/contact_model/rigid_mug_gripper \
--system_type=continuous --ts_dt=1e-3 --gripper_force=150 \
--rk_type=rk2 --rk_dt=6e-6 --accuracy=1e-3 --v_stiction_tolerance=1e-3 \
--ring_samples=8 --pad_depth=3e-3 --ring_orient=0 --ring_youngs_modulus=-1 \
--ring_dissipation=-1 --ring_static_friction=-1 --ring_dynamic_friction=-1 \
--px=0 --py=0 --pz=0 --rx=0 --ry=0 --rz=0 --playback=true \
--initial_finger_position=-0.08811
