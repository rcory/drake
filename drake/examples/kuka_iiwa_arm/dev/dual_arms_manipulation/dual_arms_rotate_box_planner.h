#pragma once

#include <memory>

#include "drake/multibody/rigid_body_tree.h"
#include "drake/multibody/parsers/model_instance_id_table.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
std::tuple<std::unique_ptr<RigidBodyTreed>, parsers::ModelInstanceIdTable> ConstructDualArmAndBox();

void VisualizePosture(RigidBodyTreed* tree, const parsers::ModelInstanceIdTable& model_instance_id_table, const Eigen::Ref<const Eigen::VectorXd>& q_kuka1, const Eigen::Ref<const Eigen::VectorXd>& q_kuka2, const Eigen::Ref<Eigen::Matrix<double, 7, 1>>& q_box);
}
}
}