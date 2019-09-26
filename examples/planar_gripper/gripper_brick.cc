#include "drake/examples/planar_gripper/gripper_brick.h"

#include "drake/common/find_resource.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace examples {
namespace planar_gripper {

std::string to_string(Finger finger) {
  switch (finger) {
    case Finger::kFinger1: {
      return "finger 1";
    }
    case Finger::kFinger2: {
      return "finger 2";
    }
    case Finger::kFinger3: {
      return "finger 3";
    }
    default:
      throw std::runtime_error("Finger not valid.");
  }
}

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
std::unique_ptr<systems::Diagram<T>> ConstructDiagram(
    multibody::MultibodyPlant<T>** plant, geometry::SceneGraph<T>** scene_graph,
    int* input_port_index_applied_spatial_force, int* output_port_index_state,
    int* input_port_index_actuation) {
  systems::DiagramBuilder<T> builder;
  std::tie(*plant, *scene_graph) =
      multibody::AddMultibodyPlantSceneGraph(&builder);
  const std::string gripper_path =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_gripper.sdf");
  multibody::Parser parser(*plant, *scene_graph);
  parser.AddModelFromFile(gripper_path, "gripper");
  examples::planar_gripper::WeldGripperFrames(*plant);
  const std::string brick_path =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_brick.sdf");
  parser.AddModelFromFile(brick_path, "brick");
  (*plant)->WeldFrames((*plant)->world_frame(),
                       (*plant)->GetFrameByName("brick_base"),
                       math::RigidTransformd());

  (*plant)->Finalize();

  AddDrakeVisualizer<T>(&builder, **scene_graph);

  *input_port_index_applied_spatial_force =
      builder.ExportInput((*plant)->get_applied_spatial_force_input_port());

  *input_port_index_actuation =
      builder.ExportInput((*plant)->get_actuation_input_port());

  *output_port_index_state =
      builder.ExportOutput((*plant)->get_state_output_port());

  // lcm::DrakeLcm lcm;
  // geometry::ConnectDrakeVisualizer(&builder, **scene_graph, &lcm);

  //// Publish contact results for visualization.
  // ConnectContactResultsToDrakeVisualizer(&builder, **plant, &lcm);

  return builder.Build();
}

template <typename T>
GripperBrickHelper<T>::GripperBrickHelper() {
  owned_diagram_ = ConstructDiagram<T>(
      &plant_, &scene_graph_, &input_port_index_applied_spatial_force_,
      &output_port_index_state_, &input_port_index_actuation_);
  diagram_ = owned_diagram_.get();
  InitializeDiagramSimulator(*diagram_);

  const geometry::SceneGraphInspector<T>& inspector =
      scene_graph_->model_inspector();
  for (int i = 0; i < 3; ++i) {
    finger_tip_sphere_geometry_ids_[i] = inspector.GetGeometryIdByName(
        plant_->GetBodyFrameIdOrThrow(
            plant_->GetBodyByName("finger" + std::to_string(i + 1) + "_link2")
                .index()),
        geometry::Role::kProximity, "gripper::link2_pad_collision");
  }
  const geometry::Shape& fingertip_shape =
      inspector.GetShape(finger_tip_sphere_geometry_ids_[0]);
  finger_tip_radius_ =
      dynamic_cast<const geometry::Sphere&>(fingertip_shape).get_radius();
  p_L2Fingertip_ = inspector.GetPoseInFrame(finger_tip_sphere_geometry_ids_[0])
                       .translation();
  const geometry::Shape& brick_shape =
      inspector.GetShape(inspector.GetGeometryIdByName(
          plant_->GetBodyFrameIdOrThrow(
              plant_->GetBodyByName("brick_link").index()),
          geometry::Role::kProximity, "brick::box_collision"));
  brick_size_ = dynamic_cast<const geometry::Box&>(brick_shape).size();

  for (int i = 0; i < 3; ++i) {
    finger_base_position_indices_[i] =
        plant_->GetJointByName("finger" + std::to_string(i + 1) + "_BaseJoint")
            .position_start();
    finger_mid_position_indices_[i] =
        plant_->GetJointByName("finger" + std::to_string(i + 1) + "_MidJoint")
            .position_start();
    finger_link2_frames_[i] =
        &(plant_->GetFrameByName("finger" + std::to_string(i + 1) + "_link2"));
  }
  brick_translate_y_position_index_ =
      plant_->GetJointByName("brick_translate_y_joint").position_start();
  brick_translate_z_position_index_ =
      plant_->GetJointByName("brick_translate_z_joint").position_start();
  brick_revolute_x_position_index_ =
      plant_->GetJointByName("brick_revolute_x_joint").position_start();
  brick_frame_ = &(plant_->GetFrameByName("brick_link"));
}

template <typename T>
const multibody::Frame<double>& GripperBrickHelper<T>::finger_link2_frame(
    Finger finger) const {
  switch (finger) {
    case Finger::kFinger1: {
      return *(finger_link2_frames_[0]);
    }
    case Finger::kFinger2: {
      return *(finger_link2_frames_[1]);
    }
    case Finger::kFinger3: {
      return *(finger_link2_frames_[2]);
    }
    default:
      throw std::invalid_argument("finger_link2_frame(), unknown finger.");
  }
}

template <typename T>
int GripperBrickHelper<T>::finger_base_position_index(Finger finger) const {
  switch (finger) {
    case Finger::kFinger1:
      return finger_base_position_indices_[0];
    case Finger::kFinger2:
      return finger_base_position_indices_[1];
    case Finger::kFinger3:
      return finger_base_position_indices_[2];
    default:
      throw std::invalid_argument(
          "finger_base_position_index(): unknown finger");
  }
}

template <typename T>
int GripperBrickHelper<T>::finger_mid_position_index(Finger finger) const {
  switch (finger) {
    case Finger::kFinger1:
      return finger_mid_position_indices_[0];
    case Finger::kFinger2:
      return finger_mid_position_indices_[1];
    case Finger::kFinger3:
      return finger_mid_position_indices_[2];
    default:
      throw std::invalid_argument(
          "finger_mid_position_index(): unknown finger");
  }
}

template <typename T>
geometry::GeometryId GripperBrickHelper<T>::finger_tip_sphere_geometry_id(
    Finger finger) const {
  switch (finger) {
    case Finger::kFinger1: {
      return finger_tip_sphere_geometry_ids_[0];
    }
    case Finger::kFinger2: {
      return finger_tip_sphere_geometry_ids_[1];
    }
    case Finger::kFinger3: {
      return finger_tip_sphere_geometry_ids_[2];
    }
    default: {
      throw std::invalid_argument(
          "finger_tip_sphere_geometry_id(): unknown finger.");
    }
  }
}

template <typename T>
multibody::CoulombFriction<T>
GripperBrickHelper<T>::GetFingerTipBrickCoulombFriction(Finger finger) const {
  const multibody::CoulombFriction<T>& brick_friction =
      plant_->default_coulomb_friction(
          plant_->GetCollisionGeometriesForBody(brick_frame().body())[0]);
  const multibody::CoulombFriction<double>& finger_tip_friction =
      plant_->default_coulomb_friction(finger_tip_sphere_geometry_id(finger));
  return multibody::CalcContactFrictionFromSurfaceProperties(
      brick_friction, finger_tip_friction);
}

// Explicit instantiation
template class GripperBrickHelper<double>;
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
