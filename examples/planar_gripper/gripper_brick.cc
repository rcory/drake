#include "drake/examples/planar_gripper/gripper_brick.h"

#include "drake/common/find_resource.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace examples {

template <typename T>
void AddDrakeVisualizer(systems::DiagramBuilder<T>*,
                        const geometry::SceneGraph<T>&) {
  // Disabling visualization for non-double scalar type T.
}

template <>
void AddDrakeVisualizer<double>(
    systems::DiagramBuilder<double>* builder,
    const geometry::SceneGraph<double>& scene_graph) {
  geometry::ConnectDrakeVisualizer(builder, scene_graph);
}

template <typename T>
void InitializeDiagramSimulator(const systems::Diagram<T>&) {}

template <>
void InitializeDiagramSimulator<double>(
    const systems::Diagram<double>& diagram) {
  systems::Simulator<double>(diagram).Initialize();
}

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
    bool add_gravity, multibody::MultibodyPlant<T>** plant,
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
      FindResourceOrThrow("drake/examples/planar_gripper/planar_brick.sdf");
  parser.AddModelFromFile(brick_path, "brick");
  (*plant)->WeldFrames((*plant)->world_frame(),
                       (*plant)->GetFrameByName("brick_base"),
                       math::RigidTransformd());

  if (add_gravity) {
  }

  (*plant)->Finalize();

  AddDrakeVisualizer<T>(&builder, **scene_graph);
  return builder.Build();
}

template <typename T>
GripperBrickSystem<T>::GripperBrickSystem(bool add_gravity) {
  diagram_ = ConstructDiagram<T>(add_gravity, &plant_, &scene_graph_);
  InitializeDiagramSimulator(*diagram_);
}

// Explicit instantiation
template class GripperBrickSystem<double>;
}  // namespace examples
}  // namespace drake
