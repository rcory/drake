#include "drake/examples/planar_gripper/planar_gripper_udp.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
namespace drake {
namespace examples {
namespace planar_gripper {
GTEST_TEST(FingerFaceAssignmentTest, Test) {
  FingerFaceAssignment dut{};
  EXPECT_EQ(dut.GetMessageSize(), sizeof(uint32_t) + sizeof(Finger) +
                                      sizeof(BrickFace) + sizeof(double) * 2);
  dut.utime = 100;
  dut.finger = Finger::kFinger1;
  dut.brick_face = BrickFace::kPosZ;
  dut.p_BoBq_B = Eigen::Vector2d(2., 3.0);

  std::vector<uint8_t> msg(dut.GetMessageSize());
  dut.Serialize(msg.data());

  FingerFaceAssignment reconstructed_dut{};
  reconstructed_dut.Deserialize(msg.data());
  EXPECT_EQ(dut, reconstructed_dut);
}

GTEST_TEST(FingerFaceAssignmentsTest, Test) {
  FingerFaceAssignments dut{2};
  EXPECT_FALSE(dut.in_contact[0]);
  EXPECT_FALSE(dut.in_contact[1]);
  dut.utime = 100;
  dut.finger_face_assignments[0].utime = dut.utime;
  dut.finger_face_assignments[0].finger = Finger::kFinger1;
  dut.finger_face_assignments[0].brick_face = BrickFace::kPosZ;
  dut.finger_face_assignments[0].p_BoBq_B = Eigen::Vector2d(2., 3.0);
  dut.finger_face_assignments[1].utime = dut.utime;
  dut.finger_face_assignments[1].finger = Finger::kFinger2;
  dut.finger_face_assignments[1].brick_face = BrickFace::kPosY;
  dut.finger_face_assignments[1].p_BoBq_B = Eigen::Vector2d(-2., -3.0);
  dut.in_contact[0] = true;
  dut.in_contact[1] = false;

  EXPECT_EQ(dut.GetMessageSize(),
            sizeof(uint32_t) + sizeof(uint32_t) +
                dut.finger_face_assignments[0].GetMessageSize() +
                dut.finger_face_assignments[1].GetMessageSize() +
                sizeof(bool) * 2);
  std::vector<uint8_t> message(dut.GetMessageSize());
  dut.Serialize(message.data());

  FingerFaceAssignments reconstructed_dut{0};
  reconstructed_dut.Deserialize(message.data());
  EXPECT_EQ(dut, reconstructed_dut);
}

GTEST_TEST(PlanarManipulandDesiredTest, Test) {
  PlanarManipulandDesired dut(3, 2);
  dut.desired_state << 0.1, 0.2, 0.3;
  dut.desired_accel << 0.4, 0.5;
  EXPECT_EQ(dut.GetMessageSize(), sizeof(uint32_t) * 3 + sizeof(double) * 5);
  std::vector<uint8_t> message(dut.GetMessageSize());
  dut.Serialize(message.data());
  PlanarManipulandDesired reconstructed_dut(0, 0);
  reconstructed_dut.Deserialize(message.data());
  EXPECT_EQ(dut, reconstructed_dut);
}

GTEST_TEST(PlanarManipulandSpatialForceTest, Test) {
  PlanarManipulandSpatialForce dut;
  dut.utime = 1001;
  dut.finger = Finger::kFinger3;
  dut.p_BoBq_B << 0.1, 0.2;
  dut.force_Bq_W << 0.3, 0.4;
  dut.torque_Bq_W = 0.5;

  EXPECT_EQ(dut.GetMessageSize(),
            sizeof(uint32_t) + sizeof(Finger) + 5 * sizeof(double));

  std::vector<uint8_t> message(dut.GetMessageSize());
  dut.Serialize(message.data());
  PlanarManipulandSpatialForce reconstructed_dut{};
  reconstructed_dut.Deserialize(message.data());
  EXPECT_EQ(dut, reconstructed_dut);

  const multibody::ExternallyAppliedSpatialForce<double> spatial_force =
      dut.ToSpatialForce(multibody::BodyIndex(1));
  EXPECT_EQ(spatial_force.body_index, multibody::BodyIndex(1));
  EXPECT_TRUE(
      CompareMatrices(spatial_force.p_BoBq_B,
                      Eigen::Vector3d(0, dut.p_BoBq_B(0), dut.p_BoBq_B(1))));
  EXPECT_TRUE(CompareMatrices(
      spatial_force.F_Bq_W.translational(),
      Eigen::Vector3d(0, dut.force_Bq_W(0), dut.force_Bq_W(1))));
  EXPECT_TRUE(CompareMatrices(spatial_force.F_Bq_W.rotational(),
                              Eigen::Vector3d(dut.torque_Bq_W, 0, 0)));
}

GTEST_TEST(PlanarManipulandSpatialForcesTest, Test) {
  PlanarManipulandSpatialForces dut(2);
  EXPECT_FALSE(dut.in_contact[0]);
  EXPECT_FALSE(dut.in_contact[1]);
  dut.utime = 100;
  dut.forces[0].utime = dut.utime;
  dut.forces[0].finger = Finger::kFinger1;
  dut.forces[0].p_BoBq_B << 0.1, 0.2;
  dut.forces[0].force_Bq_W << 0.3, 0.4;
  dut.forces[0].torque_Bq_W = 0.5;
  dut.forces[1].utime = dut.utime;
  dut.forces[1].finger = Finger::kFinger2;
  dut.forces[1].p_BoBq_B << 0.6, 0.7;
  dut.forces[1].force_Bq_W << 0.8, 0.9;
  dut.forces[1].torque_Bq_W = 1.;
  dut.in_contact[0] = true;
  dut.in_contact[1] = false;

  EXPECT_EQ(dut.GetMessageSize(),
            sizeof(uint32_t) * 2 + dut.forces[0].GetMessageSize() +
                dut.forces[1].GetMessageSize() + sizeof(bool) * 2);

  std::vector<uint8_t> message(dut.GetMessageSize());
  dut.Serialize(message.data());
  PlanarManipulandSpatialForces reconstructed_dut;
  reconstructed_dut.Deserialize(message.data());
  EXPECT_EQ(dut, reconstructed_dut);
}

GTEST_TEST(PlanarPlantStateTest, Test) {
  PlanarPlantState dut(3);
  dut.utime = 10;
  dut.plant_state << 0.1, 0.2, 0.3;
  EXPECT_EQ(dut.GetMessageSize(), 2 * sizeof(uint32_t) + sizeof(double) * 3);

  std::vector<uint8_t> message(dut.GetMessageSize());
  dut.Serialize(message.data());

  PlanarPlantState reconstructed_dut;
  reconstructed_dut.Deserialize(message.data());
  EXPECT_EQ(dut, reconstructed_dut);
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
