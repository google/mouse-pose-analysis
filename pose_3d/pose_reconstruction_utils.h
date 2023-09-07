/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_POSE_RECONSTRUCTION_UTILS_H_
#define MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_POSE_RECONSTRUCTION_UTILS_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "mouse_pose_analysis/pose_3d/kinematic_chain.h"
#include "mouse_pose_analysis/pose_3d/pose_optimizer.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "third_party/eigen3/Eigen/Core"

namespace mouse_pose {

// Exports the updated skeleton to a csv file with format:
// joint_name, joint_id, parent_id, joint_x, joint_y, joint_z, 4 x 4
// transform_mat in row major comma separated.
absl::Status ExportSkeletonCSV(
    std::string filename, const KinematicChain& chain,
    const std::vector<Eigen::Vector4d>& updated_joints,
    const std::vector<Eigen::Matrix4d>& chain_G);

// Visualizes the 2D keypoints to an input image as dots.
absl::Status Visualize2DKeypoints(std::string input_image_filename,
                                  std::string output_image_filename,
                                  const std::vector<Eigen::Vector2d>& keypoints,
                                  const Eigen::Vector3i& color,
                                  const int radius);

// Updates a kinematic chain with the joint angles (and optionally shape basis
// weights), transforms the chain with 3 rotation and 3 translation params,
// projects keypoints on the chain onto image space and return the
// transformation matrices of the keypoints if needed.
absl::Status UpdateChainAndProjectKeypoints(
    const std::vector<double>& rotation_angles,
    const Eigen::Vector3d& rigid_rotation,
    const Eigen::Vector3d& rigid_translation,
    const mouse_pose::PoseOptimizer& pose_optimizer,
    std::vector<Eigen::Vector2d>* projected_2d_keypoints,
    std::vector<Eigen::Matrix4d>* chain_transformations = nullptr,
    std::vector<float>* xyz = nullptr,
    const std::vector<double>* shape_weights = nullptr);

double MeanReprojectionError(
    const std::vector<Eigen::Vector2d>& pred_2d_keypoints,
    const std::vector<Eigen::Vector2d>& target_2d_keypoints);

cv::Mat ReadImage(std::string filename, int flags = cv::IMREAD_COLOR);

// Transforms a mesh (boh rigid and non-rigid) and writes the output to disk if
// filename is not empty.
absl::Status TransformAndSaveMesh(
    std::string filename,
    const std::vector<Eigen::Matrix4d>& chain_transformations,
    const std::vector<double>& trans_params,
    const std::vector<double>& rotate_params,
    mouse_pose::RiggedMesh* rigged_mesh);

}  // namespace mouse_pose

#endif  // MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_POSE_RECONSTRUCTION_UTILS_H_
