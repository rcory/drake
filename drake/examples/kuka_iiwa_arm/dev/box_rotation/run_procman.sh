#!/bin/bash

#export URDF=dual_iiwa14_primitive_cylinder_visual_collision.urdf
#export URDF=dual_iiwa14_primitive_cylinder_collision_only.urdf
#export URDF=dual_iiwa14_primitive_sphere_collision_only.urdf
export URDF=dual_iiwa14_primitive_sphere_visual_collision.urdf
#export URDF=dual_iiwa14_visual_only

/home/rickcory/dev/spartan/build/install/bin/bot-procman-sheriff -l iiwa_dual_box_rot.pmd
