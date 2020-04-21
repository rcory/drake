#include "drake/examples/planar_gripper/contact_force_qp.h"

#include <limits>

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/examples/planar_gripper/finger_brick.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/examples/planar_gripper/planar_gripper_utils.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/solvers/solve.h"
#include "drake/systems/framework/diagram.h"

namespace drake {
namespace examples {
namespace planar_gripper {
const double kInf = std::numeric_limits<double>::infinity();

GTEST_TEST(PlanarFingerInstantaneousQPTest, Test) {
  systems::DiagramBuilder<double> builder;

  geometry::SceneGraph<double>& scene_graph =
      *builder.AddSystem<geometry::SceneGraph>();
  scene_graph.set_name("scene_graph");

  // Make and add the planar_finger model.
  const std::string full_name =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_finger.sdf");
  multibody::MultibodyPlant<double>& plant =
      *builder.AddSystem<multibody::MultibodyPlant>(1e-3);
  multibody::Parser(&plant, &scene_graph).AddModelFromFile(full_name);
  WeldFingerFrame<double>(&plant);

  // Adds the object to be manipulated.
  auto object_file_name =
      FindResourceOrThrow("drake/examples/planar_gripper/1dof_brick.sdf");
  auto brick_index = multibody::Parser(&plant, &scene_graph)
                         .AddModelFromFile(object_file_name, "brick");
  const multibody::Frame<double>& brick_base_frame =
      plant.GetFrameByName("brick_base_link", brick_index);
  plant.WeldFrames(plant.world_frame(), brick_base_frame);

  plant.Finalize();

  const Eigen::Vector3d
      p_LtFingerTip =  // position of sphere center in tip link
      GetFingerTipSpherePositionInLt(plant, scene_graph, Finger::kFinger1);
  const double finger_tip_radius =
      GetFingerTipSphereRadius(plant, scene_graph, Finger::kFinger1);
  const Eigen::Vector3d brick_size = GetBrickSize(plant, scene_graph);
  const multibody::Frame<double>& brick_frame =
      plant.GetFrameByName("brick_link");
  const geometry::GeometryId finger_tip_geometry_id =
      GetFingertipSphereGeometryId(plant, scene_graph.model_inspector(),
                                   Finger::kFinger1);  // fingertip sphere id
  // First solve an IK problem that the finger is making contact with the brick.
  multibody::InverseKinematics ik(plant);
  // Finger in contact with +z face.
  ik.AddPositionConstraint(
      plant.GetFrameByName("finger1_tip_link"), p_LtFingerTip, brick_frame,
      Eigen::Vector3d(-kInf, -brick_size(1) / 2,
                      brick_size(2) / 2 + finger_tip_radius),
      Eigen::Vector3d(kInf, brick_size(1) / 2,
                      brick_size(2) / 2 + finger_tip_radius));

  // Add the initial brick orientation condition constraint
  const math::RotationMatrix<double> R_AbarA(
      Eigen::AngleAxisd(-M_PI_4 + .2, Eigen::Vector3d(1, 0, 0).normalized()));
  ik.AddOrientationConstraint(plant.world_frame(), R_AbarA,
                              plant.GetFrameByName("brick_link"),
                              math::RotationMatrixd(), 0);

  Eigen::Vector3d q_guess(0.1, 0.2, 0.3);
  const auto ik_result = solvers::Solve(ik.prog(), q_guess);
  EXPECT_TRUE(ik_result.is_success());
  const Eigen::Vector3d q_ik = ik_result.GetSolution(ik.q());

  // print ik results
  int bindex = plant.GetJointByName("brick_revolute_x_joint").position_start();
  int j1index = plant.GetJointByName("finger1_BaseJoint").position_start();
  int j2index = plant.GetJointByName("finger1_MidJoint").position_start();

  drake::log()->info("p_LtFingerTip: \n{}", p_LtFingerTip);
  drake::log()->info("brick_angle: {}", q_ik(bindex));
  drake::log()->info("j1_angle: {}", q_ik(j1index));
  drake::log()->info("j2_angle: {}", q_ik(j2index));

  const multibody::CoulombFriction<double>& brick_friction =
      plant.default_coulomb_friction(
          plant.GetCollisionGeometriesForBody(brick_frame.body())[0]);
  const multibody::CoulombFriction<double>& finger_tip_friction =
      plant.default_coulomb_friction(finger_tip_geometry_id);
  const double mu = multibody::CalcContactFrictionFromSurfaceProperties(
                        brick_friction, finger_tip_friction)
                        .static_friction();
  drake::log()->info("calculated mu: {}", mu);
  Eigen::Vector3d v(0.2, 0.3, -0.1);
  auto plant_context = plant.CreateDefaultContext();

  plant.SetPositions(plant_context.get(), q_ik);
  plant.SetVelocities(plant_context.get(), v);
  Eigen::Vector3d p_BFingerTip;  // fingertip sphere center in brick frame
  plant.CalcPointsPositions(*plant_context,
                            plant.GetFrameByName("finger1_tip_link"),
                            p_LtFingerTip, brick_frame, &p_BFingerTip);

  drake::log()->info("p_BFingerTip: \n{}", p_BFingerTip);

  const double theta_desired = 0.05;
  const double thetadot_desired = 0.12;
  const double thetaddot_feedforward = 0.23;
  const Eigen::Vector2d brick_state_desired(theta_desired, thetadot_desired);
  const Vector1d brick_accel_feedforward(thetaddot_feedforward);

  const double Kp = 0.1;
  const double Kd = 0.2;
  const double weight_thetaddot_error = 0.5;
  const double weight_f_Cb = 1.2;
  double I_B =
      dynamic_cast<const multibody::RigidBody<double>&>(brick_frame.body())
          .default_rotational_inertia()
          .get_moments()(0);

  const int brick_revolute_position_index =
      plant.GetJointByName("brick_revolute_x_joint").position_start();
  const double theta = q_ik(brick_revolute_position_index);
  const double thetadot = v(brick_revolute_position_index);
  double damping =
      plant.GetJointByName<multibody::RevoluteJoint>("brick_revolute_x_joint")
          .damping();
  Eigen::Vector2d p_BCb(p_BFingerTip(1), p_BFingerTip(2) - finger_tip_radius);

  Vector6<double> brick_state;
  brick_state << theta, thetadot;
  std::unordered_map<Finger, BrickFaceInfo> finger_face_assignment;
  finger_face_assignment.emplace(Finger::kFinger1,
                                 BrickFaceInfo(BrickFace::kPosZ, p_BCb, true));
  Eigen::Vector2d zero2d = Eigen::Vector2d::Zero();
  InstantaneousContactForceQP qp2(
      BrickType::PinBrick, brick_state, brick_state_desired,
      brick_accel_feedforward, zero2d, zero2d, Kp, Kd, finger_face_assignment,
      0, weight_thetaddot_error, weight_f_Cb, mu, I_B, 1, damping, 0);

  auto qp_result = solvers::Solve(qp2.prog());
  EXPECT_TRUE(qp_result.is_success());

  // Now check the result.
  // First check if the contact force is within the friction cone.
  std::unordered_map<Finger, std::pair<Eigen::Vector2d, Eigen::Vector2d>>
      finger_contact_forces = qp2.GetContactForceResult(qp_result);
  EXPECT_EQ(finger_contact_forces.size(), 1);
  Eigen::Vector2d f_Cb_B = finger_contact_forces[Finger::kFinger1].first;
  drake::log()->info("f_Cb_B_2: \n{}", f_Cb_B);
  EXPECT_LE(f_Cb_B(1), 0);
  EXPECT_LE(std::abs(f_Cb_B(0)), -mu * f_Cb_B(1));

  // Now check the cost. First compute the angular acceleration.
  double thetaddot =
      (p_BCb(0) * f_Cb_B(1) - p_BCb(1) * f_Cb_B(0) - damping * thetadot) / I_B;
  double thetaddot_des = Kp * (theta_desired - theta) +
                         Kd * (thetadot_desired - thetadot) +
                         thetaddot_feedforward;
  double cost_expected =
      weight_thetaddot_error * std::pow(thetaddot - thetaddot_des, 2) +
      weight_f_Cb * f_Cb_B.squaredNorm();
  drake::log()->info("qp2 cost: {}", qp_result.get_optimal_cost());
  EXPECT_NEAR(cost_expected, qp_result.get_optimal_cost(), 1E-9);
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
