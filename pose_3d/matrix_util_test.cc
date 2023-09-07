// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mouse_pose_analysis/pose_3d/matrix_util.h"

#include <cmath>

#include "googlemock/include/gmock/gmock.h"
#include "gtest/gtest.h"
#include "mouse_pose_analysis/pose_3d/gtest_matchers.h"
#include "third_party/eigen3/Eigen/Core"

namespace mouse_pose {
namespace {

using ::testing::Test;

TEST(KinematicChainUtilTest, SkewAxis) {
  Eigen::Vector3d axis(1., 0., 0.);
  Eigen::Vector3d center(0., 0., 0.);
  Eigen::Matrix4d twist = SkewAxis(axis, center);
  Eigen::Vector4d vertex(0., 1., 0., 1.);
  Eigen::Matrix4d rotation_mat = (0.5 * M_PI * twist).exp();
  Eigen::Vector4d rotated_vertex = rotation_mat * vertex;
  EXPECT_THAT(Eigen::Vector3d(0., 0., 1.),
              mouse_pose::test::EigenMatrixEq(rotated_vertex.head<3>()));
}

TEST(MatrixUtilTest, MatrixPowTest) {
  Eigen::Matrix4d mat(Eigen::Matrix4d::Identity());
  Eigen::Matrix4d expected(Eigen::Matrix4d::Identity());
  EXPECT_THAT(expected, mouse_pose::test::EigenMatrixEq(MatrixPow(mat, 10)));
  EXPECT_DEATH(MatrixPow(mat, -2), "");
}

TEST(MatrixUtilTest, MatrixExpTest) {
  Eigen::Vector3d rotation_axis = Eigen::Vector3d(0, 0, 1);
  Eigen::Vector3d rotation_center = Eigen::Vector3d::Zero();
  Eigen::Matrix4d mat = SkewAxis(rotation_axis, rotation_center);

  // The expected rotation matrix of rotation around z axis, 1/4 Pi degree.
  Eigen::Matrix4d expected;
  expected << 0.5 * sqrt(2.), -0.5 * sqrt(2), 0, 0, 0.5 * sqrt(2),
      0.5 * sqrt(2), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
  EXPECT_THAT(
      expected,
      mouse_pose::test::EigenMatrixNear(
          ExponentialMap(mat, static_cast<double>(M_PI * 0.25)), 1e-10));
}

// Tests the dimensions of matrix and points are correct in projection.
TEST(MatrixUtilTest, ProjectionTest) {
  Eigen::Matrix<double, 3, 4> projection_matrix;
  projection_matrix << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0;
  Eigen::Vector3d point3d;
  point3d << 3.14, 2.72, 1.41;

  Eigen::Vector2d point2d = Project3DPoint(projection_matrix, point3d);
  EXPECT_DOUBLE_EQ(point2d(0), point3d(0) / point3d(2));
  EXPECT_DOUBLE_EQ(point2d(1), point3d(1) / point3d(2));
}
}  // namespace
}  // namespace mouse_pose
