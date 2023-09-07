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

#include "mouse_pose_analysis/pose_3d/pose_reconstruction_utils.h"

#include <filesystem>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "glog/logging.h"
#include "googlemock/include/gmock/gmock.h"
#include "gtest/gtest.h"
#include "mouse_pose_analysis/pose_3d/gtest_matchers.h"
#include "mouse_pose_analysis/pose_3d/gtest_util.h"
#include "mouse_pose_analysis/pose_3d/kinematic_chain.h"
#include "mouse_pose_analysis/pose_3d/rigged_mesh_utils.h"
#include "mouse_pose_analysis/pose_3d/status.h"

namespace mouse_pose {
namespace {

using ::testing::Test;

TEST(PoseReconstructionUtilsTest, ExportSkeletonCSV) {
  std::string test_config_filename =
      GetTestRootDir() + "pose_3d/testdata/test_skeleton.csv";
  mouse_pose::KinematicChain chain;
  EXPECT_OK(chain.ReadSkeletonConfig(test_config_filename));
  EXPECT_OK(chain.ConstructKinematicChain());
  std::vector<double> alphas(3 * chain.GetJoints().size(), 0.1);
  std::vector<Eigen::Vector4d> updated_joints;
  std::vector<Eigen::Matrix4d> chain_G;
  chain.UpdateKinematicChain(alphas, &updated_joints, &chain_G);
  std::string output_filename = "/tmp/out_skeleton.csv";
  EXPECT_OK(mouse_pose::ExportSkeletonCSV(output_filename, chain,
                                           updated_joints, chain_G));
  // Check the temp output file's content.
  EXPECT_TRUE(std::filesystem::exists(output_filename));
  std::ifstream out_csv_file(output_filename);
  int record_index = 0;
  std::string line;
  while (std::getline(out_csv_file, line)) {
    if (line[0] == '#') continue;

    std::vector<std::string> tokens = absl::StrSplit(line, ',');
    EXPECT_EQ(tokens.size(), 22);
    EXPECT_EQ(tokens[0], chain.GetJointNames()[record_index]);
    Eigen::Vector2i bone;
    EXPECT_TRUE(absl::SimpleAtoi(tokens[1], &bone[0]));
    EXPECT_TRUE(absl::SimpleAtoi(tokens[2], &bone[1]));
    Eigen::Vector3d joint;
    EXPECT_TRUE(absl::SimpleAtod(tokens[3], &joint[0]));
    EXPECT_TRUE(absl::SimpleAtod(tokens[4], &joint[1]));
    EXPECT_TRUE(absl::SimpleAtod(tokens[5], &joint[2]));
    EXPECT_THAT(bone,
                mouse_pose::test::EigenMatrixEq(chain.GetBones()[record_index]));
    EXPECT_THAT(joint, mouse_pose::test::EigenMatrixNear(
                           updated_joints[record_index].head<3>(), 1e-6));
    Eigen::Matrix4d transform_mat;
    for (int i = 0; i < 16; i++) {
      EXPECT_TRUE(
          absl::SimpleAtod(tokens[i + 6], &transform_mat(i / 4, i % 4)));
    }
    EXPECT_THAT(transform_mat,
                mouse_pose::test::EigenMatrixNear(chain_G[record_index], 1e-6));
    record_index++;
  }
}

TEST(PoseReconstructionUtilsTest, Visualize2DKeypoints) {
  cv::Mat input_image_mat(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));
  std::string input_image_filename = "/tmp/test_input_image.jpg";
  std::string output_image_filename = "/tmp/test_output_image.jpg";
  EXPECT_TRUE(cv::imwrite(input_image_filename, input_image_mat));
  EXPECT_TRUE(std::filesystem::exists(input_image_filename));
  std::vector<Eigen::Vector2d> keypoints;
  keypoints.push_back(Eigen::Vector2d(1, 1));
  keypoints.push_back(Eigen::Vector2d(5, 5));
  Eigen::Vector3i color(255, 0, 0);
  EXPECT_OK(mouse_pose::Visualize2DKeypoints(
      input_image_filename, output_image_filename, keypoints, color, 3));
  cv::Mat output_image_mat = cv::imread(output_image_filename);
  EXPECT_NE(output_image_mat.data, nullptr);
  EXPECT_EQ(output_image_mat.at<cv::Vec3b>(1, 1)[0], 0);
  EXPECT_EQ(output_image_mat.at<cv::Vec3b>(1, 1)[1], 0);
  EXPECT_GT(output_image_mat.at<cv::Vec3b>(1, 1)[2], 250);
  EXPECT_EQ(output_image_mat.at<cv::Vec3b>(5, 5)[0], 0);
  EXPECT_EQ(output_image_mat.at<cv::Vec3b>(5, 5)[1], 0);
  EXPECT_GT(output_image_mat.at<cv::Vec3b>(5, 5)[2], 250);
}

}  // namespace
}  // namespace mouse_pose
