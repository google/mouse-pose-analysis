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

#include "mouse_pose_analysis/pose_3d/rigged_mesh.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "googlemock/include/gmock/gmock.h"
#include "gtest/gtest.h"
#include "mouse_pose_analysis/pose_3d/gtest_matchers.h"
#include "mouse_pose_analysis/pose_3d/gtest_util.h"
#include "mouse_pose_analysis/pose_3d/rigged_mesh_utils.h"

namespace mouse_pose {
namespace {

using ::Eigen::Matrix4d;
using ::testing::Test;

// Test fixture to prepare the obj and csv files used in the tests.
class RiggedMeshTest : public Test {
 protected:
  RiggedMeshTest() {
    // Tolerance to check floating point equality.
    // The mesh is in meters, and 1e-6 is a micrometer.
    tolerance_ = 1e-6;

    obj_filename_ = GetTestRootDir() + "pose_3d/testdata/test_tri.obj";
    weight_filename_ = GetTestRootDir() + "pose_3d/testdata/tri_weights.csv";
  }

  double tolerance_;
  std::string obj_filename_;
  std::string weight_filename_;
};

// Tests a rigged mesh can be created from files.
TEST_F(RiggedMeshTest, CreateRiggedMeshObj) {
  EXPECT_NE(nullptr,
            CreateRiggedMeshFromFiles(obj_filename_, weight_filename_));
}

// Tests the weighting is done correctly.
TEST_F(RiggedMeshTest, IdentityTransformation) {
  Matrix4d identity(Eigen::Matrix4d::Identity());
  std::vector<Matrix4d> xforms;
  xforms.push_back(identity);
  xforms.push_back(identity);

  std::unique_ptr<RiggedMesh> mesh =
      CreateRiggedMeshFromFiles(obj_filename_, weight_filename_);

  std::vector<Eigen::Vector3d> updated_vertices;
  mesh->UpdateVertices(xforms, &updated_vertices);

  // Check one entry of each vertex.
  EXPECT_DOUBLE_EQ(1, updated_vertices[0](0));
  EXPECT_DOUBLE_EQ(5, updated_vertices[1](1));
  EXPECT_DOUBLE_EQ(9, updated_vertices[2](2));
}

// Tests vertices are transformed correctly according to weights.
TEST_F(RiggedMeshTest, GlobalTranslation) {
  Matrix4d translate1(Eigen::Matrix4d::Identity());
  Matrix4d translate2(Eigen::Matrix4d::Identity());

  // Translate1 moves to the left by 1.
  translate1(0, 3) = -1;
  // Translate1 moves to the right by 1.
  translate2(0, 3) = 1;

  std::vector<Matrix4d> xforms;
  xforms.push_back(translate1);
  xforms.push_back(translate2);

  std::unique_ptr<RiggedMesh> mesh =
      CreateRiggedMeshFromFiles(obj_filename_, weight_filename_);

  std::vector<Eigen::Vector3d> updated_vertices;
  mesh->UpdateVertices(xforms, &updated_vertices);

  // First vertex moves to the right by 40%.
  EXPECT_NEAR(1 - 1 * 0.3 + 0.7, updated_vertices[0](0), tolerance_);
  // Second vertex doesn't change, since the weights to two bones are equal.
  EXPECT_NEAR(4, updated_vertices[1](0), tolerance_);
  // Third moves to the left.
  EXPECT_NEAR(7 - 1, updated_vertices[2](0), tolerance_);
}

// Tests a face is projected to a triangle of pixel value 255.
TEST_F(RiggedMeshTest, Projection) {
  std::unique_ptr<RiggedMesh> mesh =
      CreateRiggedMeshFromFiles(obj_filename_, weight_filename_);

  Eigen::Matrix<double, 3, 4> projection_matrix;
  // Focal length 18.
  projection_matrix << 18, 0, 0, 0, 0, 18, 0, 0, 0, 0, 1, 0;

  cv::Mat cvx_image = ProjectMeshToImage(*mesh, projection_matrix, 15, 20);
  Eigen::MatrixX<uint8_t> image = Eigen::Map<
      Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      cvx_image.ptr<uint8_t>(), 20, 15);

  Eigen::MatrixX<uint8_t> projected_triangle =
      Eigen::MatrixX<uint8_t>::Zero(20, 15);

  projected_triangle(12, 6) = 255;
  projected_triangle(12, 7) = 255;
  projected_triangle(13, 8) = 255;
  projected_triangle(13, 9) = 255;
  projected_triangle(14, 10) = 255;
  projected_triangle(14, 11) = 255;
  projected_triangle(15, 12) = 255;
  projected_triangle(15, 13) = 255;
  projected_triangle(16, 14) = 255;

  EXPECT_TRUE(projected_triangle.isApprox(image));
}

TEST_F(RiggedMeshTest, WriteObjectFile) {
  std::unique_ptr<RiggedMesh> mesh =
      CreateRiggedMeshFromFiles(obj_filename_, weight_filename_);
  EXPECT_OK(WriteObjFile("/tmp/foo.obj", *mesh));
}

}  // namespace
}  // namespace mouse_pose
