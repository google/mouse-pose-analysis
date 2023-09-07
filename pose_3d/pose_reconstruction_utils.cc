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

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "ceres/rotation.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "mouse_pose_analysis/pose_3d/rigged_mesh_utils.h"
#include "third_party/eigen3/Eigen/Core"

namespace mouse_pose {

absl::Status ExportSkeletonCSV(
    std::string filename, const mouse_pose::KinematicChain& chain,
    const std::vector<Eigen::Vector4d>& updated_joints,
    const std::vector<Eigen::Matrix4d>& chain_G) {
  std::ofstream csv_file(filename);
  CHECK_EQ(chain.GetJointNames().size(), chain.GetBones().size())
      << "The joint_names_ and the bones_ should have the same size.";
  CHECK_EQ(updated_joints.size(), chain.GetBones().size())
      << "The updated_joints and the bones_ should have the same size.";
  Eigen::IOFormat comma_fmt(Eigen::StreamPrecision, 0, ",", ",", "", "", "",
                            "\n");
  for (int i = 0; i < chain.GetJointNames().size(); ++i) {
    std::string line = absl::StrFormat(
        "%s,%d,%d,%f,%f,%f,", chain.GetJointNames()[i], chain.GetBones()[i][0],
        chain.GetBones()[i][1], updated_joints[i][0], updated_joints[i][1],
        updated_joints[i][2]);
    std::ostringstream matrix_str;
    matrix_str << chain_G[i].format(comma_fmt);
    line += matrix_str.str();
    csv_file << line;
    // RET_CHECK_OK(file::WriteString(csv_file, line, file::Defaults()));
  }
  return absl::OkStatus();
}

absl::Status Visualize2DKeypoints(std::string input_image_filename,
                                  std::string output_image_filename,
                                  const std::vector<Eigen::Vector2d>& keypoints,
                                  const Eigen::Vector3i& color,
                                  const int radius) {
  cv::Mat cv_image = cv::imread(input_image_filename);
  for (auto& keypoint : keypoints) {
    cv::circle(cv_image, cv::Point(keypoint[0], keypoint[1]), radius,
               CV_RGB(color[0], color[1], color[2]), -1);
  }
  // Encode and save image to file.
  cv::imwrite(output_image_filename, cv_image);
  return absl::OkStatus();
}

absl::Status UpdateChainAndProjectKeypoints(
    const std::vector<double>& rotation_angles,
    const Eigen::Vector3d& rigid_rotation,
    const Eigen::Vector3d& rigid_translation,
    const mouse_pose::PoseOptimizer& pose_optimizer,
    std::vector<Eigen::Vector2d>* projected_2d_keypoints,
    std::vector<Eigen::Matrix4d>* chain_transformations,
    std::vector<float>* xyz, const std::vector<double>* shape_weights) {
  std::vector<Eigen::Vector4d> updated_3d_keypoints;
  projected_2d_keypoints->clear();
  auto updated_joints = std::make_unique<std::vector<Eigen::Vector3d>>();
  if (shape_weights != nullptr) {
    CHECK_GT(pose_optimizer.GetKinematicChain().GetShapeBasisMean().size(), 0);
    CHECK_OK(pose_optimizer.GetKinematicChain().UpdateJointXYZLocations(
        shape_weights->data(), updated_joints.get()));
  } else {
    updated_joints.reset();
  }

  pose_optimizer.GetKinematicChain().UpdateKinematicChain(
      rotation_angles, &updated_3d_keypoints, chain_transformations,
      updated_joints.get());
  if (xyz != nullptr) {
    xyz->clear();
    xyz->resize(updated_3d_keypoints.size() * 3);
  }
  for (size_t i = 0; i < updated_3d_keypoints.size(); ++i) {
    const Eigen::Vector3d updated_keypoint = updated_3d_keypoints[i].head<3>();
    Eigen::Vector3d transformed_keypoint;
    ceres::AngleAxisRotatePoint(rigid_rotation.data(), updated_keypoint.data(),
                                transformed_keypoint.data());
    transformed_keypoint += rigid_translation;
    if (xyz != nullptr) {
      (*xyz)[i * 3] = transformed_keypoint[0];
      (*xyz)[i * 3 + 1] = transformed_keypoint[1];
      (*xyz)[i * 3 + 2] = transformed_keypoint[2];
    }
    Eigen::Vector4d augmented_keypoint;
    augmented_keypoint.block(0, 0, 3, 1) << transformed_keypoint;
    augmented_keypoint(3, 0) = 1.;
    Eigen::Vector2d projected_keypoint =
        (pose_optimizer.GetProjectionMat() * augmented_keypoint).hnormalized();
    projected_2d_keypoints->push_back(projected_keypoint);
  }
  return absl::OkStatus();
}

double MeanReprojectionError(
    const std::vector<Eigen::Vector2d>& pred_2d_keypoints,
    const std::vector<Eigen::Vector2d>& target_2d_keypoints) {
  CHECK_EQ(pred_2d_keypoints.size(), target_2d_keypoints.size());
  CHECK_GT(pred_2d_keypoints.size(), 0);
  double dist2 = 0;
  for (int i = 0; i < pred_2d_keypoints.size(); ++i) {
    dist2 += (target_2d_keypoints[i] - pred_2d_keypoints[i]).squaredNorm();
  }
  return sqrt(dist2) / pred_2d_keypoints.size();
}

// Reads an image from GFile paths.
cv::Mat ReadImage(std::string filename, int flags) {
  std::ifstream image_file(filename);
  CHECK(image_file) << "Can't open " << filename;
  std::string content;
  image_file.seekg(0, image_file.end);
  int len = image_file.tellg();
  image_file.seekg(0, image_file.beg);

  std::vector<char> buf(len);
  image_file.read(buf.data(), len);
  return cv::imdecode(buf, flags);
}

absl::Status TransformAndSaveMesh(
    std::string filename,
    const std::vector<Eigen::Matrix4d>& chain_transformations,
    const std::vector<double>& trans_params,
    const std::vector<double>& rotate_params,
    mouse_pose::RiggedMesh* rigged_mesh) {
  // Transform mesh based on the optimized rigid transformation parameters.
  rigged_mesh->DeformMesh(chain_transformations);
  rigged_mesh->TransformMesh(
      Eigen::Vector3f(rotate_params[0], rotate_params[1], rotate_params[2]),
      Eigen::Vector3f(trans_params[0], trans_params[1], trans_params[2]));
  if (filename.empty()) {
    return absl::OkStatus();
  }
  return WriteObjFile(filename, *rigged_mesh);
}
}  // namespace mouse_pose
