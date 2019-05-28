#include "drake/examples/planar_gripper/gripper_brick.h"

#include "drake/common/find_resource.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace examples {
template <typename T>
void WeldGripperFrames(multibody::MultibodyPlant<T>* plant) {
  // This function is copied and adapted from planar_gripper_simulation.py
  const double outer_radius = 0.19;
  const double f1_angle = M_PI / 3;
  const math::RigidTransformd XT(math::RollPitchYaw<double>(f1_angle, 0, 0),
                                 Eigen::Vector3d(0, 0, outer_radius));

  // Weld the first finger.
  math::RigidTransformd X_PC1(math::RollPitchYaw<double>(f1_angle, 0, 0),
                              Eigen::Vector3d::Zero());
  X_PC1 = X_PC1 * XT;
  const multibody::Frame<T>& finger1_base_frame =
      plant->GetFrameByName("finger1_base");
  plant->WeldFrames(plant->world_frame(), finger1_base_frame, X_PC1);

  // Weld the second finger.
  const math::RigidTransformd X_PC2 =
      math::RigidTransformd(math::RollPitchYawd(M_PI / 3 * 2, 0, 0),
                            Eigen::Vector3d::Zero()) *
      X_PC1;
  const multibody::Frame<T>& finger2_base_frame =
      plant->GetFrameByName("finger2_base");
  plant->WeldFrames(plant->world_frame(), finger2_base_frame, X_PC2);

  // Weld the 3rd finger.
  const math::RigidTransformd X_PC3 =
      math::RigidTransformd(math::RollPitchYawd(M_PI / 3 * 2, 0, 0),
                           Eigen::Vector3d::Zero()) *
      X_PC2;
  const multibody::Frame<T>& finger3_base_frame =
      plant->GetFrameByName("finger3_base");
  plant->WeldFrames(plant->world_frame(), finger3_base_frame, X_PC3);
}

template <typename T>
std::unique_ptr<systems::Diagram<T>> ConstructDiagram(
    multibody::MultibodyPlant<T>** plant,
    geometry::SceneGraph<T>** scene_graph) {
  systems::DiagramBuilder<T> builder;
  std::tie(*plant, *scene_graph) =
      multibody::AddMultibodyPlantSceneGraph(&builder);
  const std::string gripper_path =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_gripper.sdf");
  multibody::Parser parser(*plant, *scene_graph);
  parser.AddModelFromFile(gripper_path, "gripper");
  WeldGripperFrames(*plant);
  const std::string brick_path =
      FindResourceOrThrow("drake/examples/planar_gripper/1dof_brick.sdf");
  parser.AddModelFromFile(brick_path, "brick");

  (*plant)->Finalize();
  return builder.Build();
}

template <typename T>
GripperBrickSystem<T>::GripperBrickSystem() {
  diagram_ = ConstructDiagram<T>(&plant_, &scene_graph_);
}

// Explicit instantiation
template class GripperBrickSystem<double>;
}  // namespace examples
}  // namespace drake
