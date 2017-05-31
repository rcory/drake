#pragma once

#include "drake/common/drake_assert.h"
#include "drake/common/drake_path.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_common.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_tree.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
std::unique_ptr<RigidBodyTreed> ConstructDualArmAndBox();
}
}
}